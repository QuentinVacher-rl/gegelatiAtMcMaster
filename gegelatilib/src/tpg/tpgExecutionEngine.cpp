/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019 - 2022) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019 - 2022)
 * Thomas Bourgoin <tbourgoi@insa-rennes.fr> (2021)
 *
 * GEGELATI is an open-source reinforcement learning framework for training
 * artificial intelligence based on Tangled Program Graphs (TPGs).
 *
 * This software is governed by the CeCILL-C license under French law and
 * abiding by the rules of distribution of free software. You can use,
 * modify and/ or redistribute the software under the terms of the CeCILL-C
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty and the software's author, the holder of the
 * economic rights, and the successive licensors have only limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading, using, modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean that it is complicated to manipulate, and that also
 * therefore means that it is reserved for developers and experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and, more generally, to use and operate it in the
 * same conditions as regards security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-C license and that you accept its terms.
 */

#include <algorithm>
#include <set>
#include <vector>
#include <cmath>  

#include "program/programExecutionEngine.h"
#include "tpg/tpgEdge.h"
#include "tpg/tpgTeam.h"

#include "tpg/tpgExecutionEngine.h"

void TPG::TPGExecutionEngine::setArchive(Archive* newArchive)
{
    this->archive = newArchive;
}

void TPG::TPGExecutionEngine::setErrorWeights(const std::map<Program::Program*, std::vector<double>>* newErrorWeights)
{
    progExecutionEngine.setErrorWeights(newErrorWeights);
}

Environment TPG::TPGExecutionEngine::getEnvironment()
{
    return this->env;
}

void TPG::TPGExecutionEngine::applyActivationFunctionOnActions(std::vector<double>& actionsTaken)
{

    for(int i = 0; i < actionsTaken.size(); i++){
        if(std::isnan(actionsTaken[i])){
            actionsTaken[i] = -std::numeric_limits<double>::infinity();
        }
    }

    // Sigmoid function
    if(env.getParams().activationFunction == "sigmoid"){
        for(size_t i=0; i<actionsTaken.size(); i++){
            actionsTaken[i] = 1.0 / (1.0 + std::exp(-actionsTaken[i]));
        }
    } else if(env.getParams().activationFunction == "tanh"){
        std::transform(actionsTaken.begin(), actionsTaken.end(), actionsTaken.begin(), [](double x) { return std::tanh(x); });

    } else if(env.getParams().activationFunction == "none"){
        for (double& actionTaken : actionsTaken) {
            actionTaken = std::clamp(actionTaken, -1.0, 1.0);
        }
    } else {
        throw std::runtime_error("Activation function for converting continuous actions not known");
    }
}

void TPG::TPGExecutionEngine::resetAllMemoryRegisters()
{
    this->progExecutionEngine.resetAllMemoryRegisters();
    this->progExecutionEngine.resetSharedRegisters();
}

double TPG::TPGExecutionEngine::evaluateEdge(const TPGEdge& edge)
{
    // Get the program
    Program::Program& prog = edge.getProgram();

    // Set the progExecutionEngine to the program
    this->progExecutionEngine.setProgram(prog);

    // Execute the program.
    double result = this->progExecutionEngine.executeProgram();

    // Filter NaN results: replace with -inf
    result = (std::isnan(result)) ? -std::numeric_limits<double>::infinity()
                                  : result;

    // Put the result in the archive before returning it.
    if (this->archive != NULL) {
        this->archive->addRecording(&prog, progExecutionEngine.getDataSources(),
                                    result);
    }

    return result;
}

bool TPG::TPGExecutionEngine::executeAction(
    const TPGVertex* currentAction, std::vector<double>* actionsTaken)
{

    auto action = (const TPGAction*)(currentAction);
    // Save the action value if the action ID is choosen for the first
    // time.
    if(env.getNbContinuousActions() == 0){
        (*actionsTaken)[action->getActionClass()] = (double)action->getActionID();

    } else if (env.getParams().mutation.tpg.multiActionProg) {

        for(auto edge: currentAction->getOutgoingEdges()){
            auto actionEdge = dynamic_cast<TPGActionEdge*>(edge);

            // Set the current action class for shared registers
            this->progExecutionEngine.setActionClass(actionEdge->getActionClass());

            // Evaluate the edge and set the action value
            (*actionsTaken)[actionEdge->getActionClass()] = this->evaluateEdge(*actionEdge);

            // If activate, save the shared value
            if(env.getParams().isActionSharedMem && env.getParams().nbSharedRegisters > 0){
                progExecutionEngine.setSharedRegisterValues(actionEdge->getProgram(), actionEdge->getActionClass());
            }
        }

    } else {

        auto edge = *currentAction->getOutgoingEdges().begin();

        this->evaluateEdge(*edge);


        // Set the current action class for shared registers
        this->progExecutionEngine.setActionClass(0);

        auto result = this->progExecutionEngine.getRegisterValues(edge->getProgramSharedPointer(), this->getEnvironment().getNbContinuousActions());

        actionsTaken->assign(result.begin(), result.end());

        if(env.getParams().isActionSharedMem && env.getParams().nbSharedRegisters > 0){
            progExecutionEngine.setSharedRegisterValues(edge->getProgram(), 0);
        }


    }
    return true;


}

std::vector<const TPG::TPGEdge*> TPG::TPGExecutionEngine::executeTeam(
    const TPGVertex* currentTeam,
    std::vector<const TPGVertex*>& visitedVertices,
    std::vector<double>* actionsTaken, uint64_t nbEdgesActivated)
{

    std::vector<const TPGEdge*> traversedEdges;

    std::set<uint64_t> actionAssessed;

    // Add current team to the visited vertices.
    visitedVertices.push_back(currentTeam);

    // Copy outgoing edge list.
    const std::list<TPG::TPGEdge*>& outgoingEdges =
        currentTeam->getOutgoingEdges();


    // Calcul the bids of all teams.
    std::vector<std::pair<TPG::TPGEdge*, double>> resultsBid;
    for (auto edge : outgoingEdges) {
        
        // Set the current action class for shared registers
        this->progExecutionEngine.setActionClass(*edge->getDestination()->getAssessedActions().begin());
        // Calcul program bid.
        double bid = this->evaluateEdge(*edge);
        resultsBid.push_back(std::make_pair(edge, bid));
    };

    // Sorting with ">=" is not possible, a sort with "<" then reverse is used
    // instead. Sort the results.
    std::sort(resultsBid.begin(), resultsBid.end(),
              [](const std::pair<TPG::TPGEdge*, double>& a,
                 const std::pair<TPG::TPGEdge*, double>& b) {
                  return a.second < b.second;
              });

    // Reverse
    std::reverse(resultsBid.begin(), resultsBid.end());

    size_t i = 0;
    // For all TPGEdge evaluated.
    while (i < resultsBid.size() && actionAssessed.size() < actionsTaken->size()) {

        // Get the pair with the edge and the bid.
        auto destination = resultsBid[i].first->getDestination();

        //std::cout<<"Team "<<currentTeam<<"  | Case "<<i<<"  | Bid "<<resultsBid[i].second<<"  | ";

        if(!destination->hasSameAssessedActions(actionAssessed)){

            /*std::cout<<"Choose for actions :";
            for(auto a: destination->getAssessedActions()){
                std::cout<<a<<"-";
            }std::cout<<std::endl;*/

            if(env.getParams().nbSharedRegisters > 0){
                for(auto actionClass: destination->getAssessedActions()){
                    progExecutionEngine.setSharedRegisterValues(resultsBid[i].first->getProgram(), actionClass);
                }
            }

            // If edge destination is an action
            if (dynamic_cast<const TPGAction*>(destination)) {
                executeAction(destination, actionsTaken);


                // Add the action the the visited vertices and the edge to the
                // traversed edges.
                visitedVertices.push_back(destination);
                traversedEdges.push_back(resultsBid[i].first);

            }
            else {

                // Only if the team has not already been visited.
                if (std::find(visitedVertices.begin(), visitedVertices.end(),
                            destination) == visitedVertices.end()) {

                    // Add the edge to the traversed edges.
                    traversedEdges.push_back(resultsBid[i].first);
                    // If edge destination is a team, launch recursively the method.

                    //std::cout<<"Executing team :"<< destination<<std::endl;
                    executeTeam((const TPGTeam*)(destination), visitedVertices,
                                actionsTaken, nbEdgesActivated);
                }
            }

            auto destinationAssessedActions = destination->getAssessedActions();
            // Add the action of the observed vertices to the action used
            actionAssessed.insert(destinationAssessedActions.begin(), destinationAssessedActions.end());
        } else {

            /*std::cout<<"Ignore with actions :";
            for(auto a: destination->getAssessedActions()){
                std::cout<<a<<"-";
            }std::cout<<std::endl;*/
        }


        i++;
    }

    return traversedEdges;
}

std::pair<std::vector<const TPG::TPGVertex*>, std::vector<double>> TPG::
    TPGExecutionEngine::executeFromRoot(
        const TPGVertex& root, const std::vector<uint64_t>& initActions,
        uint64_t nbEdgesActivated)
{

    // Reset the shared memory
    if(!env.getParams().useMemoryRegisters){
        progExecutionEngine.resetSharedRegisters();
    }

    const TPGVertex* currentVertex = &root;
    std::vector<const TPGVertex*> visitedVertices;

    // An action value must be positive, so -1 for an action mean that no action
    // value is choosen yet.
    std::vector<double> actionsTaken(env.getNbContinuousActions(), 0.0);

    // Execute the team only if it is really a team
    if (dynamic_cast<const TPGTeam*>(&root)) {
        /*std::cout<<"START ROOT "<<currentVertex<<"   ";
        for(auto a: currentVertex->getAssessedActions()){
            std::cout<<a<<"-";
        }std::cout<<std::endl;*/
        executeTeam(dynamic_cast<const TPGTeam*>(currentVertex),
                    visitedVertices, &actionsTaken, nbEdgesActivated);
    }
    else {
        executeAction(currentVertex, &actionsTaken);
    }


    // If discrete action are used, browse the raw list of actions and replace the "-1" action by the initial
    // value.
    if(this->getEnvironment().getNbContinuousActions() > 0){
        if(nbEdgesActivated != 1){
            throw std::runtime_error("The number of edges activable can not be different to 1 in this mode");
        }

        this->applyActivationFunctionOnActions(actionsTaken);

    }

    auto results = std::make_pair(visitedVertices, actionsTaken);

    return results;
}
