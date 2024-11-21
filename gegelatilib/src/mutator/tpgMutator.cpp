/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019 - 2022) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019 - 2022)
 * Nicolas Sourbier <nsourbie@insa-rennes.fr> (2020)
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
#include <mutex>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

#include "archive.h"

#include "program/programExecutionEngine.h"
#include "tpg/tpgAction.h"
#include "tpg/tpgEdge.h"
#include "tpg/tpgActionEdge.h"
#include "tpg/tpgGraph.h"
#include "tpg/tpgTeam.h"

#include "mutator/mutationParameters.h"
#include "mutator/programMutator.h"
#include "mutator/rng.h"
#include "mutator/tpgMutator.h"



void Mutator::TPGMutator::initRandomTPG(
    TPG::TPGGraph& graph, const Mutator::MutationParameters& params,
    Mutator::RNG& rng, const std::vector<uint64_t>& vectActions)
{
    uint64_t nbActions = graph.getEnvironment().getNbContinuousActions();

    if (params.tpg.nbActionEdgeInit > graph.getEnvironment().getNbContinuousActions()){
        throw std::runtime_error("Maximum initial number of outgoing edges "
                            "cannot exceed the number of action");
    }
    if (params.tpg.maxInitOutgoingEdges < 1 ) {
        throw std::runtime_error("Maximum initial number of outgoing edges "
                                 "cannot exceed the number of action");
    }

    if (params.tpg.initNbActions < params.tpg.initNbRoots){
        throw std::runtime_error("Can't be more root team than action");
    }
    /*if (nbActions < 2) {
        throw std::runtime_error("A TPG with a single action makes no sense.");
    }*/
    //if (params.tpg.initNbRoots < nbActions) {
    //    throw std::runtime_error("The number of init roots should be above or "
    //                             "equal to the number of actions.");
    //}
    if(vectActions.size() >1){
        throw std::runtime_error("No more than one action class for now");
    }
    // Empty graph
    graph.clear();

    // Create teams, programs and Actions
    std::vector<const TPG::TPGAction*> actions;
    std::vector<const TPG::TPGTeam*> teams;
    std::vector<std::shared_ptr<Program::Program>> contextPrograms;
    std::vector<std::shared_ptr<Program::Program>> actionPrograms;


    for (size_t i = 0; i < params.tpg.initNbActions; i++) {

        // Create an action
        actions.push_back(&(graph.addNewAction(0, 0)));
        
        std::set<uint64_t> actionUsed;
        for(size_t j = 0; j < params.tpg.nbActionEdgeInit; j++){

            // Create an action program
            actionPrograms.emplace_back(new Program::Program(graph.getEnvironment(), true));
            // RandomInit the Programs
            Mutator::ProgramMutator::initRandomProgram(*actionPrograms.back(), params,
                                                        rng);

            // Find an action class not already used by this action 
            uint64_t actionClass;
            do{
                actionClass = rng.getInt32(0, nbActions - 1);}
            while(actionUsed.find(actionClass) != actionUsed.end());
            actionUsed.insert(actionClass);

            // Create the action edge
            graph.addNewActionEdge(*actions.at(i),
                                actionPrograms.back(),
                                actionClass);
        }

        graph.orderActionEdges(actions.back());
    }

    for(size_t i = 0; i < params.tpg.initNbRoots; i++){

        // Create a team
        teams.push_back(&(graph.addNewTeam()));   

        // Create a program to connect the team and the action
        contextPrograms.emplace_back(new Program::Program(graph.getEnvironment(), false));
        // RandomInit the Programs
        Mutator::ProgramMutator::initRandomProgram(*contextPrograms.back(), params,
                                                   rng);

        // Connect the team to the action
        graph.addNewEdge(*teams.at(i),
                         *actions.at(i),
                         contextPrograms.at(i));
    }

    graph.updateAllAssessedActions();

    // Add additional connections to TPG
    // Team-by-Team
    for (const TPG::TPGTeam* team : teams) {
        // Pick a number of additional outedge
        size_t nbAdditionalEdges =
            rng.getUnsignedInt64(0, params.tpg.maxInitOutgoingEdges - 1);

        // For each additional edge to add
        for (uint64_t i = 0; i < nbAdditionalEdges; i++) {
            // Pick 2 random programs not already used by the Team
            int64_t randomContextProgIndex[2] = {-1, -1};
            int pickedContextProgram = 0;
            {
                // Copy the list of programs
                std::vector<int> availableChoices(contextPrograms.size());
                std::iota(availableChoices.begin(), availableChoices.end(), 0);
                // Remove already connected ones
                auto iter = availableChoices.begin();
                while (iter < availableChoices.end()) {
                    if (std::count_if(
                            team->getOutgoingEdges().begin(),
                            team->getOutgoingEdges().end(),
                            [&iter, &contextPrograms](const TPG::TPGEdge* edge) {
                                return &edge->getProgram() ==
                                       contextPrograms.at(*iter).get();
                            }) > 0) {
                        iter = availableChoices.erase(iter);
                    }
                    else {
                        iter++;
                    }
                }

                // Pick two programs (if possible, maybe only one is available)
                for (int i = 0; i < 2 && availableChoices.size() > 0; i++) {
                    uint64_t progNr =
                        rng.getUnsignedInt64(0, availableChoices.size() - 1);
                    randomContextProgIndex[i] = availableChoices.at(progNr);
                    availableChoices.erase(availableChoices.begin() + progNr);
                    pickedContextProgram++;
                }
            }
            // Select the least used program for the connection
            uint64_t selectedContextProgramIndex =
                (pickedContextProgram > 1 &&
                 contextPrograms.at(randomContextProgIndex[1]).use_count() <
                     contextPrograms.at(randomContextProgIndex[0]).use_count())
                    ? randomContextProgIndex[1]
                    : randomContextProgIndex[0];


            // Copy the programs
            contextPrograms.emplace_back(new Program::Program(*contextPrograms.at(selectedContextProgramIndex).get()));

            auto actionIndex = rng.getInt32(0, actions.size() - 1);

            graph.addNewEdge(*team,
                             *actions.at(actionIndex),
                             contextPrograms.back());


        }

        graph.updateAssessedActions(team);
    }

}



void Mutator::TPGMutator::removeRandomActionEdge(TPG::TPGGraph& graph,
                                           const TPG::TPGAction& action,
                                           Mutator::RNG& rng)
{
    // Pick an outgoing edge randomly,
    const std::list<TPG::TPGEdge*>& pickableEdges = action.getOutgoingEdges();

    // Note: No need to take special care of Actions. Since cycles can not
    // appear in TPG with the current mutation process, there is no need to
    // maintain an action within each team.

    // Pick a random edge
    auto iterSet = pickableEdges.begin();
    std::advance(iterSet, rng.getUnsignedInt64(0, pickableEdges.size() - 1));
    const TPG::TPGEdge* removedEdge = *iterSet;
    graph.removeActionEdge(*removedEdge);
}

void Mutator::TPGMutator::addRandomActionEdge(
    TPG::TPGGraph& graph, const TPG::TPGAction& action,
    const std::list<const TPG::TPGEdge*>& preExistingActionEdges, Mutator::RNG& rng)
{
    // Pick an edge (excluding ones from the team and edges with the team as a
    // destination)
    auto pickableEdges(preExistingActionEdges);
    // cf erase-remove idiom
    pickableEdges.erase(
        std::remove_if(
            pickableEdges.begin(), pickableEdges.end(),
            [&action](const TPG::TPGEdge* edge) -> bool {
                if(action.getAssessedActions().find(dynamic_cast<const TPG::TPGActionEdge*>(edge)->getActionClass()) ==
                   action.getAssessedActions().end())
                {
                    return edge->getSource() == &action;
                } else {
                    return true;
                }
                   
            }
        ),
        pickableEdges.end()
    );

    if(pickableEdges.size() == 0){
        // Chances are really low but the pickableEdges can be empty
        return;
    }

    // Pick a pickable Edge
    // (This code assumes that the set of pickable edge is never empty..
    // otherwise it will throw an exception. Possible solution if needed
    // initialize an entirely new program and pick a random target.)
    std::list<const TPG::TPGEdge*>::iterator iter = pickableEdges.begin();
    std::advance(iter, rng.getUnsignedInt64(0, pickableEdges.size() - 1));
    const TPG::TPGEdge* pickedEdge = *iter;



    // Create new edge from team and with the same ProgramSharedPointer
    // But with the team as its source
    // throw std::runtime_error if the edge is not from the graph;
    const TPG::TPGEdge& newEdge = graph.cloneEdge(*pickedEdge);
    graph.setEdgeSource(newEdge, action);
}




void Mutator::TPGMutator::swapActionEdges(            
    TPG::TPGGraph& graph, const TPG::TPGAction& action, Mutator::RNG& rng)
{

    // Randomly select two edges
    size_t index1 = rng.getUnsignedInt64(0, action.getOutgoingEdges().size() - 1);
    size_t index2 = rng.getUnsignedInt64(0, action.getOutgoingEdges().size() - 2);
    if(index1 == index2){
        index2++;
    }

    // Use a single iterator to traverse and identify both edges
    TPG::TPGEdge* edge1 = nullptr;
    TPG::TPGEdge* edge2 = nullptr;
    size_t currentIndex = 0;

    for (auto it = action.getOutgoingEdges().begin(); it != action.getOutgoingEdges().end(); ++it, ++currentIndex) {
        if (currentIndex == index1) {
            edge1 = *it;
        } else if (currentIndex == index2) {
            edge2 = *it;
        }
        if (edge1 && edge2) {
            break; // Stop as soon as both edges are found
        }
    }

    // Extract and swap action classes
    auto actionClass1 = dynamic_cast<TPG::TPGActionEdge*>(edge1)->getActionClass();
    auto actionClass2 = dynamic_cast<TPG::TPGActionEdge*>(edge2)->getActionClass();

    graph.setActionClassEdge(edge1, actionClass2);
    graph.setActionClassEdge(edge2, actionClass1);

}



void Mutator::TPGMutator::mutateTPGActionEdge(
    TPG::TPGGraph& graph, const TPG::TPGAction& action, TPG::TPGActionEdge* actionEdge,
    std::list<std::shared_ptr<Program::Program>>& newPrograms,
    const Mutator::MutationParameters& params, Mutator::RNG& rng)
{ 

        
    // copy program
    newPrograms.push_back(actionEdge->getProgramSharedPointer());

    // Change action ID randomly if the action do not contain all actions. 
    if (action.getAssessedActions().size() < graph.getEnvironment().getNbContinuousActions() &&
        params.tpg.pChangeActionClass > rng.getDouble(0.0, 1.0)) {

        uint64_t newActionID = rng.getInt32(0, graph.getEnvironment().getNbContinuousActions() - 1);
        while(action.getAssessedActions().find(newActionID) != action.getAssessedActions().end()){
            newActionID = rng.getInt32(0, graph.getEnvironment().getNbContinuousActions() - 1);
        }

        actionEdge->setActionClass(newActionID);   
        
        graph.updateAssessedActions(&action);   

    }

}


void Mutator::TPGMutator::mutateTPGAction(
    TPG::TPGGraph& graph, const TPG::TPGAction& action,
    const std::vector<const TPG::TPGAction*>& preExistingActions,
    std::list<const TPG::TPGEdge*> preExistingActionEdges,
    std::list<std::shared_ptr<Program::Program>>& newPrograms,
    const Mutator::MutationParameters& params, Mutator::RNG& rng)
{ 

    // 1. Remove randomly selected edges
    // Keep at least two edges (otherwise the team is useless)
    double proba = params.tpg.pActionEdgeDeletion;
    while (action.getOutgoingEdges().size() > 1 &&
        proba > rng.getDouble(0.0, 1.0)) {
        removeRandomActionEdge(graph, action, rng);

        // Decrement the proba of removing another edge
        proba *= params.tpg.pActionEdgeDeletion;

        // Update assessed actions    
        graph.updateAssessedActions(&action);
        
    }
    

    // 2. Add random duplicated edge with the team as its source
    proba = params.tpg.pActionEdgeAddition;
    while (action.getOutgoingEdges().size() < graph.getEnvironment().getNbContinuousActions() &&
           proba > rng.getDouble(0.0, 1.0)) {

        // Add an edge (by duplication of an existing one)
        addRandomActionEdge(graph, action, preExistingActionEdges, rng);

        // Decrement the proba of adding another edge
        proba *= params.tpg.pActionEdgeAddition;

        // Update assessed actions    
        graph.updateAssessedActions(&action);
                
    }
    



    // 3. swap randomly selected edges
    // With at least two edges
    proba = params.tpg.pSwapActionProgram;
    while (action.getOutgoingEdges().size() > 2 &&
            proba > rng.getDouble(0.0, 1.0)) {
        swapActionEdges(graph, action, rng);

        // Decrement the proba of swapping two edges
        proba *= params.tpg.pSwapActionProgram;

        
    }


    bool anyMutationDone = false;
    do {
        std::vector<uint64_t> indexUsed;
        uint64_t index;
        // 4. mutate randomly selected program on action Edge. 
        proba = params.tpg.pMutateActionProgram;
        while(indexUsed.size() < action.getOutgoingEdges().size()  && proba > rng.getDouble(0.0, 1.0)){
            
            do {
                index = rng.getUnsignedInt64(0, action.getOutgoingEdges().size()-1);
            } while(std::find(indexUsed.begin(), indexUsed.end(), index) != indexUsed.end()) ;

            indexUsed.push_back(index);
    
            std::list<TPG::TPGEdge *>::const_iterator iter = action.getOutgoingEdges().begin();
            std::advance(iter, index);
            TPG::TPGActionEdge* actionEdge = dynamic_cast<TPG::TPGActionEdge*>(*iter);

            mutateTPGActionEdge(graph, action, actionEdge, newPrograms, params, rng);

            proba *= params.tpg.pMutateActionProgram;

            anyMutationDone = true;
        }
    } while (!anyMutationDone);


    graph.orderActionEdges(&action);

}


void Mutator::TPGMutator::removeRandomEdge(TPG::TPGGraph& graph,
                                           const TPG::TPGTeam& team,
                                           Mutator::RNG& rng)
{
    // Pick an outgoing edge randomly,
    const std::list<TPG::TPGEdge*>& pickableEdges = team.getOutgoingEdges();

    // Note: No need to take special care of Actions. Since cycles can not
    // appear in TPG with the current mutation process, there is no need to
    // maintain an action within each team.

    // Pick a random edge
    auto iterSet = pickableEdges.begin();
    std::advance(iterSet, rng.getUnsignedInt64(0, pickableEdges.size() - 1));
    const TPG::TPGEdge* removedEdge = *iterSet;
    graph.removeEdge(*removedEdge);
}

void Mutator::TPGMutator::addRandomEdge(
    TPG::TPGGraph& graph, const TPG::TPGTeam& team,
    const std::list<const TPG::TPGEdge*>& preExistingEdges, Mutator::RNG& rng)
{
    // Pick an edge (excluding ones from the team and edges with the team as a
    // destination)
    auto pickableEdges(preExistingEdges);
    // cf erase-remove idiom
    pickableEdges.erase(
        std::remove_if(
            pickableEdges.begin(), pickableEdges.end(),
            [&team](const TPG::TPGEdge* edge) -> bool {
                return edge->getSource() == &team ||
                    edge->getDestination() == &team;
            }
        ),
        pickableEdges.end()
    );

    // Pick a pickable Edge
    // (This code assumes that the set of pickable edge is never empty..
    // otherwise it will throw an exception. Possible solution if needed
    // initialize an entirely new program and pick a random target.)
    std::list<const TPG::TPGEdge*>::iterator iter = pickableEdges.begin();
    std::advance(iter, rng.getUnsignedInt64(0, pickableEdges.size() - 1));
    const TPG::TPGEdge* pickedEdge = *iter;



    // Create new edge from team and with the same ProgramSharedPointer
    // But with the team as its source
    // throw std::runtime_error if the edge is not from the graph;
    const TPG::TPGEdge& newEdge = graph.cloneEdge(*pickedEdge);
    graph.setEdgeSource(newEdge, team);
}


void Mutator::TPGMutator::mutateOutgoingEdge(
    TPG::TPGGraph& graph, const TPG::TPGEdge* edge, const TPG::TPGTeam& team,
    const std::vector<const TPG::TPGTeam*>& preExistingTeams,
    const std::vector<const TPG::TPGAction*>& preExistingActions,
    std::list<std::shared_ptr<Program::Program>>& newPrograms,
    const Mutator::MutationParameters& params, Mutator::RNG& rng)
{


    // Edge target modification
    // As it Stephen kelly's work, Edge target modification is conditionned
    // to the modification of the prealable Edge.Program behavior.
    if (rng.getDouble(0.0, 1.0) < params.tpg.pEdgeDestinationChange) {
        // Pick an edge among preexisting vertices
        const TPG::TPGVertex* target = NULL;


        bool targetAction = rng.getDouble(0, 1) < params.tpg.pEdgeDestinationIsAction;

        if (targetAction) {
            target = preExistingActions.at(
                rng.getUnsignedInt64(0, preExistingActions.size() - 1)); 
        } else {
            target = preExistingTeams.at(
                rng.getUnsignedInt64(0, preExistingTeams.size() - 1));
        }


        // Change the target
        // Changing the target should not fail.
        graph.setEdgeDestination(*edge, *target);
    } 
}

void Mutator::TPGMutator::mutateTPGTeam(
    TPG::TPGGraph& graph, const Archive& archive, const TPG::TPGTeam& team,
    const std::vector<const TPG::TPGTeam*>& preExistingTeams,
    const std::vector<const TPG::TPGAction*>& preExistingActions,
    const std::list<const TPG::TPGEdge*>& preExistingEdges,
    const std::list<const TPG::TPGEdge*>& preExistingActionEdges,
    std::list<std::shared_ptr<Program::Program>>& newPrograms,
    const Mutator::MutationParameters& params, Mutator::RNG& rng)
{


    // 1. Remove randomly selected edges
    {
        // Keep at least two edges (otherwise the team is useless)
        double proba = params.tpg.pEdgeDeletion;
        while (team.getOutgoingEdges().size() > 1 &&
            proba > rng.getDouble(0.0, 1.0)) {
            removeRandomEdge(graph, team, rng);

            // Decrement the proba of removing another edge
            proba *= params.tpg.pEdgeDeletion;

            // Update assessed actions    
            graph.updateAssessedActions(&team);
                    

            
        }
    }

    // 2. Add random duplicated edge with the team as its source
    {
        double proba = params.tpg.pEdgeAddition;
        while (team.getOutgoingEdges().size() < params.tpg.maxOutgoingEdges &&
            proba > rng.getDouble(0.0, 1.0)) {

            // Add an edge (by duplication of an existing one)
            addRandomEdge(graph, team, preExistingEdges, rng);

            // Decrement the proba of adding another edge
            proba *= params.tpg.pEdgeAddition;

            // Update assessed actions    
            graph.updateAssessedActions(&team);
                    
        }
    }
    // 3. Mutate edges of the team
    {
        bool anyMutationDone = false;
        do {
            // Process edge-by-edge
            // And possibly modify their target
            for (TPG::TPGEdge* edge : team.getOutgoingEdges()) {

                // Mutate the program, but need to choose between context program or action program if it exist
                if (rng.getDouble(0.0, 1.0) < params.tpg.pProgramMutation){

                    // If destination is action, and probability win, mutate the action program
                    if(dynamic_cast<const TPG::TPGAction*>(edge->getDestination()) != nullptr &&
                        rng.getDouble(0.0, 1.0) > params.tpg.probaContextOverActionProgram){

                        // Clone the randomly selected action
                        const TPG::TPGAction& newAction = (const TPG::TPGAction&)graph.cloneVertex(*edge->getDestination());

                        // Mutate the action
                        mutateTPGAction(graph, newAction, preExistingActions, preExistingActionEdges,
                                        newPrograms, params, rng);

                        // Set the action
                        graph.setEdgeDestination(*edge, newAction);

                    } else {
                        // Mutate the context program
                        // Add it to the list of new Program to be mutated.
                        newPrograms.push_back(edge->getProgramSharedPointer());

                        mutateOutgoingEdge(graph, edge, team, preExistingTeams,
                                        preExistingActions, newPrograms, params,
                                        rng);
                    }

                    // Update assessed actions    
                    graph.updateAssessedActions(&team);
                    
                    anyMutationDone = true;
                }
            }
        } while (!anyMutationDone);
    }




}
     
void Mutator::TPGMutator::mutateProgramBehaviorAgainstArchive(
    std::shared_ptr<Program::Program>& newProg,
    const Mutator::MutationParameters& params, const Archive& archive,
    Mutator::RNG& rng)
{
    // If the Program behavior should be new after mutation:
    std::shared_ptr<Program::Program> newProgCopy(nullptr);
    if (params.tpg.forceProgramBehaviorChangeOnMutation) {
        // Copy the program to check that its behavior is changed before
        // verifying its unicity against the archive
        newProgCopy = std::make_shared<Program::Program>(*newProg);
    }

    bool allUnique;
    // Mutate behavior until it changes (against the archive).
    do {

        const ProgramParameters& progParams = newProg->isActionProgram() ? params.actProg : params.contProg;

        // If a new program is created
        if (rng.getDouble(0.0, 1.0) < progParams.pNewProgram) {
            Mutator::ProgramMutator::initRandomProgram(*newProg, params, rng);
        }
        else {
            // Mutate until something is mutated (i.e. the function returns
            // true) And until the program behavior is changed
            while (!(
                Mutator::ProgramMutator::mutateProgram(*newProg, params, rng) &&
                !(newProgCopy != nullptr &&
                  newProg->hasIdenticalBehavior(*newProgCopy))))
                ;
        }
        // Check for uniqueness in archive
        auto archivedDataHandlers = archive.getDataHandlers();
        std::map<size_t, double> hashesAndResults;
        Program::ProgramExecutionEngine pee(*newProg);
        for (std::pair<
                 size_t,
                 std::vector<std::reference_wrapper<const Data::DataHandler>>>
                 archiveDatahandler : archivedDataHandlers) {
            // Execute the mutated program on the archive data handlers
            pee.setDataSources(archiveDatahandler.second);
            double result = pee.executeProgram();
            hashesAndResults.insert({archiveDatahandler.first, result});
        }

        // If the result is not unique, do another mutation.
        allUnique = archive.areProgramResultsUnique(hashesAndResults);
    } while (!allUnique);
}

void Mutator::TPGMutator::mutateNewProgramBehaviors(
    const uint64_t& maxNbThreads,
    std::list<std::shared_ptr<Program::Program>>& newPrograms,
    Mutator::RNG& rng, const Mutator::MutationParameters& params,
    const Archive& archive)
{
    // This is a computing intensive part of the mutation process
    // Hence the parallelization.
    if (maxNbThreads <= 1) {
        // Sequential (kept for determinism check mostly)
        for (std::shared_ptr<Program::Program> newProg : newPrograms) {
            Mutator::RNG privateRNG(rng.getUnsignedInt64(0, UINT64_MAX));
            mutateProgramBehaviorAgainstArchive(newProg, params, archive,
                                                privateRNG);
        }
    }
    else {
        // Parallel
        // Create job list with Program pointers and seed
        std::queue<std::pair<std::shared_ptr<Program::Program>, uint64_t>>
            programsToMutate;
        for (std::shared_ptr<Program::Program> newProg : newPrograms) {
            programsToMutate.push(
                {newProg, rng.getUnsignedInt64(0, UINT64_MAX)});


        }

        std::mutex mutexMutation;

        // Function executed in threads
        auto parallelWorker = [&programsToMutate, &mutexMutation, &params,
                               &archive]() {
            Mutator::RNG privateRNG;
            // While there is work to be done
            bool jobDone;
            do {
                std::pair<std::shared_ptr<Program::Program>, uint64_t> job;
                jobDone = false;
                { // get one job critical section
                    std::lock_guard lock(mutexMutation);
                    if (programsToMutate.size() != 0) {
                        jobDone = true;
                        job = programsToMutate.front();
                        programsToMutate.pop();
                    }
                }

                //  Do the job (if any)
                if (jobDone) {
                    privateRNG.setSeed(job.second);
                    mutateProgramBehaviorAgainstArchive(job.first, params,
                                                        archive, privateRNG);
                }
            } while (jobDone);
        };

        // Start threads
        std::vector<std::thread> threads;
        for (auto idx = 0; idx < maxNbThreads - 1; idx++) {
            threads.emplace_back(std::thread(parallelWorker));
        }

        // Work in the main thread also
        parallelWorker();

        // Join the threads
        for (auto& thread : threads) {
            thread.join();
        }
    }
}

void Mutator::TPGMutator::populateTPG(TPG::TPGGraph& graph,
                                      const Archive& archive,
                                      const Mutator::MutationParameters& params,
                                      Mutator::RNG& rng,
                                      const std::vector<uint64_t>& vectActions,
                                      uint64_t maxNbThreads)
{   
    // Get current vertex set (copy)
    auto vertices(graph.getVertices());
    // Get current root teams (copy)
    auto rootVertices(graph.getRootVertices());
    // Get root Teams
    std::vector<const TPG::TPGTeam*> rootTeams;
    std::for_each(rootVertices.begin(), rootVertices.end(),
                  [&rootTeams](const TPG::TPGVertex* vertex) {
                      if (dynamic_cast<const TPG::TPGTeam*>(vertex) !=
                          nullptr) {
                          rootTeams.push_back((const TPG::TPGTeam*)vertex);
                      }
                  });

    // If the graph doesn't contain any root teams, call the init procedure.
    // (note that execution of this code is not a very good sign.. maybe an
    // exception would be more appropriate?)
    if (rootVertices.size() == 0) {
        initRandomTPG(graph, params, rng, vectActions);
        vertices = graph.getVertices();
        rootVertices = graph.getRootVertices();
        rootTeams.clear();
        std::for_each(rootVertices.begin(), rootVertices.end(),
                      [&rootTeams](const TPG::TPGVertex* vertex) {
                          rootTeams.push_back((const TPG::TPGTeam*)vertex);
                      });
    }

    // Pre compute liste of available TPGTeam and TPGActions
    std::vector<const TPG::TPGTeam*> preExistingTeams;
    std::vector<const TPG::TPGAction*> preExistingActions;

    std::for_each(
        vertices.begin(), vertices.end(),
        [&preExistingActions, &preExistingTeams](const TPG::TPGVertex* target) {
            if (dynamic_cast<const TPG::TPGAction*>(target) != nullptr) {
                preExistingActions.push_back((const TPG::TPGAction*)target);
            }
            else {
                preExistingTeams.push_back((const TPG::TPGTeam*)target);
            }
        });

    // Get a list of pre existing edges before mutations (copy)
    std::list<const TPG::TPGEdge*> preExistingEdges;
    std::for_each(
        graph.getEdges().begin(), graph.getEdges().end(),
        [&preExistingEdges](const std::unique_ptr<TPG::TPGEdge>& edge) {
            preExistingEdges.push_back(edge.get());
        });

    // Get a list of pre existing action Edges before mutations (copy)
    std::list<const TPG::TPGEdge*> preExistingActionEdges;
    std::for_each(
        graph.getActionEdges().begin(), graph.getActionEdges().end(),
        [&preExistingActionEdges](const std::unique_ptr<TPG::TPGEdge>& edge) {
            preExistingActionEdges.push_back(edge.get());
        });

    // Create an empty list to store Programs to mutate.
    std::list<std::shared_ptr<Program::Program>> newPrograms;

    int nbActionsBefore = 0;
    for(auto vertex: graph.getVertices()){
        if (dynamic_cast<const TPG::TPGAction*>(vertex) != nullptr){
            nbActionsBefore++;
        }
    }

    // While the target is not reached, add new teams
    uint64_t currentNumberOfRoot = rootVertices.size();

    auto roots = graph.getRootVertices();
    uint64_t currentNumberOfActionRoot = std::count_if(roots.begin(), roots.end(),
        [](const TPG::TPGVertex* roots) {
            return dynamic_cast<const TPG::TPGAction*>(roots) != nullptr;
        });
    uint64_t currentNumberOfTeamRoot = std::count_if(roots.begin(), roots.end(),
        [](const TPG::TPGVertex* roots) {
            return dynamic_cast<const TPG::TPGTeam*>(roots) != nullptr;
        });


    if(params.tpg.proportionActionRoots + params.tpg.proportionTeamRoots > 1){
        throw std::runtime_error("Too many proportion!");
    }


    uint64_t nbActionsWanted = params.tpg.proportionActionRoots * (double)params.tpg.nbRoots;
    uint64_t nbTeamWanted = params.tpg.proportionTeamRoots * (double)params.tpg.nbRoots;

    while (params.tpg.nbRoots > currentNumberOfRoot) {

        // Select a random existing root
        uint64_t clonedRootIndex =
            rng.getUnsignedInt64(0, rootVertices.size() - 1);

        if(dynamic_cast<const TPG::TPGTeam*>(rootVertices.at(clonedRootIndex)) != nullptr){

            if(currentNumberOfTeamRoot < nbTeamWanted || currentNumberOfActionRoot >= nbActionsWanted){
                // clone it (the vertex and all its outgoing edges)
                const TPG::TPGTeam& newTeam = (const TPG::TPGTeam&)graph.cloneVertex(
                    *rootVertices.at(clonedRootIndex));
                // Apply mutations to the root
                mutateTPGTeam(graph, archive, newTeam, preExistingTeams,
                            preExistingActions, preExistingEdges, preExistingActionEdges, newPrograms, params,
                            rng);
            }


        } else {

            if(currentNumberOfActionRoot < nbActionsWanted || currentNumberOfTeamRoot >= nbTeamWanted){

                // clone it (the vertex and all its outgoing edges)
                const TPG::TPGAction& newAction = (const TPG::TPGAction&)graph.cloneVertex(
                    *rootVertices.at(clonedRootIndex));
                // Apply mutations to the root
                mutateTPGAction(graph, newAction, preExistingActions, preExistingActionEdges,
                                newPrograms, params, rng);
            }
        }


        // Check the new number of roots
        // Needed since preExisting root may be subsumed by new ones.
        currentNumberOfRoot = graph.getNbRootVertices();
        roots = graph.getRootVertices();
        currentNumberOfActionRoot = std::count_if(roots.begin(), roots.end(),
            [](const TPG::TPGVertex* roots) {
                return dynamic_cast<const TPG::TPGAction*>(roots) != nullptr;
            });
        currentNumberOfTeamRoot = std::count_if(roots.begin(), roots.end(),
            [](const TPG::TPGVertex* roots) {
                return dynamic_cast<const TPG::TPGTeam*>(roots) != nullptr;
            });
    }

    int nbActionsafter = 0;
    for(auto vertex: graph.getVertices()){
        if (dynamic_cast<const TPG::TPGAction*>(vertex) != nullptr){
            nbActionsafter++;
        }
    }

    mutateNewProgramBehaviors(maxNbThreads, newPrograms, rng, params, archive);

}


std::map<Program::Line*, std::vector<double>> Mutator::TPGMutator::generateErrorWeights(
    TPG::TPGGraph& graph, const Mutator::MutationParameters& params, Mutator::RNG& rng, double xmin, double xmax)
{
    // Initialise the map
    std::map<Program::Line*, std::vector<double>> errorWeights;

    std::vector<const std::list<std::unique_ptr<TPG::TPGEdge>>*> allEdges;
    allEdges.push_back(&graph.getEdges());
    allEdges.push_back(&graph.getActionEdges());

    for (const auto* edgeList : allEdges) {
        // Assurez-vous que edgeList est un pointeur vers une liste de unique_ptr
        for (const auto& edge : *edgeList) {
            
            Program::Program* program = &edge->getProgram();

            for(size_t idx_line = 0; idx_line< program->getNbLines(); idx_line++){

                Program::Line* line = &program->getLine(idx_line);

                if(line->getNbConstants() > 0 && !program->isIntron(idx_line)){

                    // Initialise the vector of errors of the program
                    std::vector<double> errorThisLine(line->getNbConstants());

                    std::generate(errorThisLine.begin(), errorThisLine.end(), [&rng, &xmin, xmax]() {
                        return rng.getDouble(xmin, xmax); // TODO NORMAL DISTRIBUTION
                    });

                    errorWeights.insert(std::make_pair(line, errorThisLine));
                }

            }


        }  
    }


    return errorWeights;
}

std::map<Program::Line*, std::vector<double>> Mutator::TPGMutator::generateTwinNegErrorWeights(
    TPG::TPGGraph& graph, std::map<Program::Line*, std::vector<double>> initError)
{
    // Initialise the map
    std::map<Program::Line*, std::vector<double>> twinNegErrorWeights;

    for(auto pair: initError){
        std::vector<double> negatedError(pair.first->getNbConstants());
        std::transform(pair.second.begin(), pair.second.end(), negatedError.begin(),
                [](double x) { return -x; });
        
        twinNegErrorWeights.insert(std::make_pair(pair.first, negatedError));  
    }

    return twinNegErrorWeights;
   
}

