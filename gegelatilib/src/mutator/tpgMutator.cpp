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

    if (params.tpg.maxInitOutgoingEdges > graph.getEnvironment().getNbContinuousActions()) {
        throw std::runtime_error("Maximum initial number of outgoing edges "
                                 "cannot exceed the number of action");
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

    for (size_t i = 0; i < params.tpg.initNbRoots; i++) {
        teams.push_back(&(graph.addNewTeam()));        
        actions.push_back(&(graph.addNewAction(0, 0)));
        actionPrograms.emplace_back(new Program::Program(graph.getEnvironment(), true));
        // RandomInit the Programs
        Mutator::ProgramMutator::initRandomProgram(*actionPrograms.back(), params,
                                                    rng);
        contextPrograms.emplace_back(new Program::Program(graph.getEnvironment(), false));
        // RandomInit the Programs
        Mutator::ProgramMutator::initRandomProgram(*contextPrograms.back(), params,
                                                   rng);
    }


    // Connect each team with two distinct actions, through two distinct
    // programs Association here are determinists since randomness would
    // uselessly complicate the code while bringing no real value since anyway,
    // Programs have been initialized randomly.
    for (size_t i = 0; i < params.tpg.initNbRoots; i++) {
        graph.addNewActionEdge(*actions.at(i),
                               actionPrograms.at(i),
                               (i % nbActions));

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


            // Pick 2 random programs not already used by the Team
            int64_t randomActionProgIndex[2] = {-1, -1};
            int pickedActionProgram = 0;
            {
                // Copy the list of programs
                std::vector<int> availableChoices(actionPrograms.size());
                std::iota(availableChoices.begin(), availableChoices.end(), 0);
                // Remove already connected ones
                auto iter = availableChoices.begin();
                while (iter < availableChoices.end()) {
                    if (std::count_if(
                            team->getOutgoingEdges().begin(),
                            team->getOutgoingEdges().end(),
                            [&iter, &actionPrograms](const TPG::TPGEdge* edge) {
                                return &edge->getProgram() ==
                                       actionPrograms.at(*iter).get();
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
                    randomActionProgIndex[i] = availableChoices.at(progNr);
                    availableChoices.erase(availableChoices.begin() + progNr);
                    pickedActionProgram++;
                }
            }
            // Select the least used program for the connection
            uint64_t selectedActionProgramIndex =
                (pickedActionProgram > 1 &&
                 actionPrograms.at(randomActionProgIndex[1]).use_count() <
                     actionPrograms.at(randomActionProgIndex[0]).use_count())
                    ? randomActionProgIndex[1]
                    : randomActionProgIndex[0];


            // Copy the programs
            contextPrograms.emplace_back(new Program::Program(*contextPrograms.at(selectedContextProgramIndex).get()));
            actionPrograms.emplace_back(new Program::Program(*actionPrograms.at(selectedActionProgramIndex).get()));


            actions.push_back(&(graph.addNewAction(0, 0)));

            uint64_t actiondestination = 0;
            // Be sure to not choose an already selected action
            if(!params.tpg.allowSameActionInTeam){
                actiondestination = rng.getInt32(0, nbActions - 1 - team->getAssessedActions().size());

                while(team->getAssessedActions().find(actiondestination) != team->getAssessedActions().end()){
                    actiondestination++;
                }

            } else {
                actiondestination = rng.getInt32(0, nbActions - 1);
            }

            // Add the connections
            graph.addNewActionEdge(*actions.back(), 
                                   actionPrograms.back(), 
                                   actiondestination);

            graph.addNewEdge(*team,
                             *actions.back(),
                             contextPrograms.back());

            graph.updateAssessedActions(actions.back());

        }
    }

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
    const std::list<const TPG::TPGEdge*>& preExistingEdges, Mutator::RNG& rng, const Mutator::MutationParameters& params)
{
    // Pick an edge (excluding ones from the team and edges with the team as a
    // destination)
    auto pickableEdges(preExistingEdges);
    // cf erase-remove idiom
    pickableEdges.erase(
        std::remove_if(
            pickableEdges.begin(), pickableEdges.end(),
            [&team, &params](const TPG::TPGEdge* edge) -> bool {
                return edge->getSource() == &team ||
                    edge->getDestination() == &team ||
                    (!params.tpg.allowSameActionInTeam &&
                        team.hasSameAssessedActions(edge->getDestination()->getAssessedActions()));
            }
        ),
        pickableEdges.end()
    );

    if(pickableEdges.empty()){
        // Sometimes, with really bad luck, no edges with different assessed actions are available.
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
    graph.setEdgeSource(newEdge, team);
}

void Mutator::TPGMutator::mutateTPGAction(
    TPG::TPGGraph& graph, const TPG::TPGAction& action, const TPG::TPGTeam& team,
    const std::vector<const TPG::TPGAction*>& preExistingActions,
    std::list<std::shared_ptr<Program::Program>>& newPrograms,
    const Mutator::MutationParameters& params, Mutator::RNG& rng)
{ 

    if(action.getOutgoingEdges().size() > 1){
        throw std::runtime_error("should be only one action edge per action vertex for now");
    }

    auto actionEdge = dynamic_cast<TPG::TPGActionEdge*>(*action.getOutgoingEdges().begin());

    newPrograms.push_back(actionEdge->getProgramSharedPointer());

    // Change action ID randomlu. 
    if (params.tpg.pChangeActionClass > rng.getDouble(0.0, 1.0)) {

        // Do not try to add an edge if the team contain all action and multiple one are not allowed. 
        if(!params.tpg.allowSameActionInTeam && 
            team.getAssessedActions().size() == graph.getEnvironment().getNbContinuousActions()){
                
                return;
        }

        

        uint64_t newActionID = rng.getInt32(0, graph.getEnvironment().getNbContinuousActions() - 1);
        while(team.getAssessedActions().find(newActionID) != team.getAssessedActions().end() && 
              !params.tpg.allowSameActionInTeam){
            newActionID = rng.getInt32(0, graph.getEnvironment().getNbContinuousActions() - 1);
        }

        actionEdge->setActionClass(newActionID);   
        
        graph.updateAssessedActions(&action);         
    }



}

void Mutator::TPGMutator::mutateEdgeDestination(
    TPG::TPGGraph& graph, const TPG::TPGEdge* edge, const TPG::TPGTeam& team,
    const std::vector<const TPG::TPGTeam*>& preExistingTeams,
    const std::vector<const TPG::TPGAction*>& preExistingActions,
    std::list<std::shared_ptr<Program::Program>>& newPrograms,
    const Mutator::MutationParameters& params, Mutator::RNG& rng)
{
    // Pick an edge among preexisting vertices
    const TPG::TPGVertex* target = NULL;
    bool targetAction;

    if(params.tpg.allowSameActionInTeam){

        targetAction = rng.getDouble(0, 1) < params.tpg.pEdgeDestinationIsAction;

        if (targetAction) {
            target = preExistingActions.at(
                rng.getUnsignedInt64(0, preExistingActions.size() - 1)); 
        } else {
            target = preExistingTeams.at(
                rng.getUnsignedInt64(0, preExistingTeams.size() - 1));
        }
    } else {
        // Determine if the new target should be an action or a team
        // First check that an action is available

        std::vector<const TPG::TPGTeam*> pickeableTeams;
        std::vector<const TPG::TPGAction*> pickeableActions;

        // Get Pickable teams and actions
        for (const auto* actionExi : preExistingActions) {
            if (!team.hasSameAssessedActions(actionExi->getAssessedActions())) {
                pickeableActions.push_back(actionExi);
            }
        }
        for (const auto* teamExi : preExistingTeams) {
            if (!team.hasSameAssessedActions(teamExi->getAssessedActions())) {
                pickeableTeams.push_back(teamExi);
            }
        }

        if (!pickeableActions.empty() && !pickeableTeams.empty()) {
            targetAction = rng.getDouble(0, 1) < params.tpg.pEdgeDestinationIsAction;
            // No target possible
        } else if(!pickeableActions.empty() && pickeableTeams.empty()){
            targetAction = true;
        } else if(pickeableActions.empty() && !pickeableTeams.empty()){
            targetAction = false;
        } else {
            return;
        }

        // Pick any target
        // Note: Having an action in all teams is no longer enforced,
        // as the presence of cycle in TPGs is not possible according to the current
        // mutation process.
        if (targetAction) {
            target = pickeableActions.at(
                rng.getUnsignedInt64(0, pickeableActions.size() - 1)); 
        } else {
            target = pickeableTeams.at(
                rng.getUnsignedInt64(0, pickeableTeams.size() - 1));
        }
    }





    // Change the target
    // Changing the target should not fail.
    graph.setEdgeDestination(*edge, *target);

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
        mutateEdgeDestination(graph, edge, team, preExistingTeams, preExistingActions,
                              newPrograms, params, rng);
    } 
}

void Mutator::TPGMutator::mutateTPGTeam(
    TPG::TPGGraph& graph, const Archive& archive, const TPG::TPGTeam& team,
    const std::vector<const TPG::TPGTeam*>& preExistingTeams,
    const std::vector<const TPG::TPGAction*>& preExistingActions,
    const std::list<const TPG::TPGEdge*>& preExistingEdges,
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

            // Do not try to add an edge if the team contain all action and multiple one are not allowed. 
            if(!params.tpg.allowSameActionInTeam && 
                team.getAssessedActions().size() == graph.getEnvironment().getNbContinuousActions()){
                    break;
            }

            // Add an edge (by duplication of an existing one)
            addRandomEdge(graph, team, preExistingEdges, rng, params);

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
                        mutateTPGAction(graph, newAction, team, preExistingActions,
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

    // Creer complete list
    std::list<const TPG::TPGEdge*> preExistingAllEdges;
    preExistingAllEdges.insert(preExistingAllEdges.end(), preExistingEdges.begin(), preExistingEdges.end());
    preExistingAllEdges.insert(preExistingAllEdges.end(), preExistingActionEdges.begin(), preExistingActionEdges.end());


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



    while (params.tpg.nbRoots > currentNumberOfRoot) {

        // Select a random existing root
        uint64_t clonedRootIndex =
            rng.getUnsignedInt64(0, rootVertices.size() - 1);

        if(dynamic_cast<const TPG::TPGTeam*>(rootVertices.at(clonedRootIndex)) != nullptr){

                // clone it (the vertex and all its outgoing edges)
                const TPG::TPGTeam& newTeam = (const TPG::TPGTeam&)graph.cloneVertex(
                    *rootVertices.at(clonedRootIndex));
                // Apply mutations to the root
                mutateTPGTeam(graph, archive, newTeam, preExistingTeams,
                            preExistingActions, preExistingEdges, newPrograms, params,
                            rng);


        } else {
            throw std::runtime_error("Action should never be root");
        }


        // Check the new number of roots
        // Needed since preExisting root may be subsumed by new ones.
        currentNumberOfRoot = graph.getNbRootVertices();

    }

    int nbActionsafter = 0;
    for(auto vertex: graph.getVertices()){
        if (dynamic_cast<const TPG::TPGAction*>(vertex) != nullptr){
            nbActionsafter++;
        }
    }

    mutateNewProgramBehaviors(maxNbThreads, newPrograms, rng, params, archive);

}


std::map<Program::Program*, std::vector<double>> Mutator::TPGMutator::generateErrorWeights(
    TPG::TPGGraph& graph, const Mutator::MutationParameters& params, Mutator::RNG& rng, double xmin, double xmax)
{
    // Initialise the map
    std::map<Program::Program*, std::vector<double>> errorWeights;

    std::vector<const std::list<std::unique_ptr<TPG::TPGEdge>>*> allEdges;
    allEdges.push_back(&graph.getEdges());
    allEdges.push_back(&graph.getActionEdges());

    for (const auto* edgeList : allEdges) {
        // Assurez-vous que edgeList est un pointeur vers une liste de unique_ptr
        for (const auto& edge : *edgeList) {
            Program::Program* program = &edge->getProgram();

            // Initialise the vector of errors of the program
            std::vector<double> errorThisProgram(program->getNbConstants());

            std::generate(errorThisProgram.begin(), errorThisProgram.end(), [&rng, &xmin, xmax]() {
                return rng.getDouble(xmin, xmax); // TODO NORMAL DISTRIBUTION
            });

            errorWeights.insert(std::make_pair(program, errorThisProgram));
        }  
    }


    return errorWeights;
}

std::map<Program::Program*, std::vector<double>> Mutator::TPGMutator::generateTwinNegErrorWeights(
    TPG::TPGGraph& graph, std::map<Program::Program*, std::vector<double>> initError)
{
    // Initialise the map
    std::map<Program::Program*, std::vector<double>> twinNegErrorWeights;

    for(auto pair: initError){
        std::vector<double> negatedError(pair.first->getNbConstants());
        std::transform(pair.second.begin(), pair.second.end(), negatedError.begin(),
                [](double x) { return -x; });
        
        twinNegErrorWeights.insert(std::make_pair(pair.first, negatedError));  
    }

    return twinNegErrorWeights;
   
}

