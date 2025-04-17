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

#include "program/programExecutionEngine.h"
#include "tpg/tpgEdge.h"

#include "tpg/tpgExecutionEngine.h"

void TPG::TPGExecutionEngine::setArchive(Archive* newArchive)
{
    this->archive = newArchive;
}

void TPG::TPGExecutionEngine::applyActivationFunctionOnActions(
    std::vector<double>& actionsTaken)
{

    for (int i = 0; i < actionsTaken.size(); i++) {
        if (std::isnan(actionsTaken[i])) {
            actionsTaken[i] = -std::numeric_limits<double>::infinity();
        }
    }

    // Sigmoid function
    if (env.getParams().activationFunction == "sigmoid") {
        for (size_t i = 0; i < actionsTaken.size(); i++) {
            actionsTaken[i] = 1.0 / (1.0 + std::exp(-actionsTaken[i]));
        }
    }
    else if (env.getParams().activationFunction == "tanh") {
        std::transform(actionsTaken.begin(), actionsTaken.end(),
                       actionsTaken.begin(),
                       [](double x) { return std::tanh(x); });
    }
    else if (env.getParams().activationFunction == "none") {
        for (double& actionTaken : actionsTaken) {
            actionTaken = std::clamp(actionTaken, -1.0, 1.0);
        }
    }
    else {
        throw std::runtime_error(
            "Activation function for converting continuous actions not known");
    }
}

double TPG::TPGExecutionEngine::evaluateEdge(const TPGEdge& edge, const TPGAgent& agent)
{
    // Get the program
    Program::Program& prog = agent.getProgram(edge);

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

const TPG::TPGEdge& TPG::TPGExecutionEngine::evaluateDecisionVertex(const TPGDecisionVertex& team, const TPGAgent& agent)
{
    // Copy outgoing edge list
    const std::list<TPG::TPGEdge*>& outgoingEdges = team.getOutgoingEdges();

    // Note: No need to exclude previously visited edges as the graph is now
    // assumed to be acyclic.


    // Evaluate all TPGEdge
    // First
    TPGEdge* bestEdge = *outgoingEdges.begin();
    double bestBid = this->evaluateEdge(*bestEdge, agent);
    // Others
    for (auto iter = ++outgoingEdges.begin(); iter != outgoingEdges.end();
         iter++) {
            TPGEdge* edge = *iter;
        double bid = this->evaluateEdge(*edge, agent);
        if (bid >= bestBid) {
            bestEdge = edge;
            bestBid = bid;
        }
    }

    return *bestEdge;
}

const std::pair<std::vector<const TPG::TPGVertex*>, std::vector<double>> TPG::
    TPGExecutionEngine::executeFromRoot(
        const TPGAgent& agent, const std::vector<uint64_t>& initActions)
{



    const TPGVertex* currentActivationVertex = agent.getRootSpecies();
    const TPGVertex* currentDecisionVertex;
    const TPGEdge* edge = nullptr;

    std::vector<const TPGVertex*> visitedVertices;
    visitedVertices.push_back(currentActivationVertex);

    // An action value must be positive, so -1 for an action mean that no action
    // value is choosen yet.
    std::vector<double> actionsTaken(env.getNbContinuousActions(), 0.0);

    // Vector of the decisionVertex to evaluate
    std::vector<const TPG::TPGVertex*> decisionVertexToEvaluate;

    // Get the actions of the init root and the decision vertex to evaluate.
    for(auto edge: currentActivationVertex->getOutgoingEdges()){
        if (dynamic_cast<TPG::TPGActionEdge*>(edge)){
            auto actionEdge = dynamic_cast<TPGActionEdge*>(edge);
            // Evaluate the edge and set the action value
            actionsTaken[actionEdge->getActionClass()] = this->evaluateEdge(*edge, agent);
        } else if(dynamic_cast<TPG::TPGConnectionEdge*>(edge)) {
            decisionVertexToEvaluate.push_back(edge->getDestination());
        } else {
            throw std::runtime_error("This vertex should be a activation vertex, with no DecisionEdge");
        }
    }

    while (decisionVertexToEvaluate.size() != 0){

        // Get the first element in the decision vertex to evaluate and erase it from the list
        currentDecisionVertex = decisionVertexToEvaluate.front();
        decisionVertexToEvaluate.erase(decisionVertexToEvaluate.begin());
        visitedVertices.push_back(currentDecisionVertex);

        // Get the next edge
        edge = &this->evaluateDecisionVertex(*(const TPGDecisionVertex*)currentDecisionVertex, agent);

        // update currentActivationVertex and backup in visitedVertex.
        currentActivationVertex = edge->getDestination();
        visitedVertices.push_back(currentActivationVertex);

        // Get the actions of the init root and the decision vertex to evaluate.
        for(auto edge: currentActivationVertex->getOutgoingEdges()){
            if (dynamic_cast<TPG::TPGActionEdge*>(edge)){
                auto actionEdge = dynamic_cast<TPGActionEdge*>(edge);
                // Evaluate the edge and set the action value
                actionsTaken[actionEdge->getActionClass()] = this->evaluateEdge(*edge, agent);
            } else if(dynamic_cast<TPG::TPGConnectionEdge*>(edge)) {
                decisionVertexToEvaluate.push_back(edge->getDestination());
            } else {
                throw std::runtime_error("This vertex should be a activation vertex, with no DecisionEdge");
            }
        }
    }




    this->applyActivationFunctionOnActions(actionsTaken);
    

    return std::make_pair(visitedVertices, actionsTaken);

}
