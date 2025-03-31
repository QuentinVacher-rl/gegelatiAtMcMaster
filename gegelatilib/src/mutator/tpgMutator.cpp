/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019 - 2024) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019 - 2022)
 * Nicolas Sourbier <nsourbie@insa-rennes.fr> (2020)
 * Quentin Vacher <qvacher@insa-rennes.fr> (2024)
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
#include <array>

#include "archive.h"

#include "program/programExecutionEngine.h"
#include "tpg/tpgEdge.h"
#include "tpg/tpgGraph.h"
#include <tpg/tpgActivationVertex.h>
#include <tpg/tpgDecisionVertex.h>

#include "mutator/mutationParameters.h"
#include "mutator/programMutator.h"
#include "mutator/rng.h"
#include "mutator/tpgMutator.h"

void Mutator::TPGMutator::initRandomTPG(
    TPG::TPGGraph& graph, const Mutator::MutationParameters& params,
    Mutator::RNG& rng, uint64_t nbActions)
{
    // Number of action edge per action vertex.
    uint64_t nbActionEdgeInit = params.tpg.nbActionEdgeInit;


    if (nbActionEdgeInit > graph.getEnvironment().getNbContinuousActions()){
        throw std::runtime_error("Maximum initial number of outgoing action edges "
                            "cannot exceed the number of action");
    }

    if (nbActionEdgeInit == 0){
        throw std::runtime_error("Initial number of outgoing action edges "
                            "should not be 0");
    }

    // If no error but case with continuous actions, nbActions is set to the
    // number of action vertex created
    nbActions = params.tpg.initNbActions;
    

    // Empty graph
    graph.clear();

    // Create teams, programs and Actions
    std::vector<const TPG::TPGActivationVertex*> teams;
    std::vector<std::shared_ptr<Program::Program>> programs;

    for (size_t i = 0; i < params.tpg.initNbActions; i++) {
        teams.push_back(&(graph.addNewActivationVertex()));

        for(size_t j = 0; j < nbActionEdgeInit; j++){
            
            // Create a program and specify action program
            std::shared_ptr<Program::Program> p =
                std::make_shared<Program::Program>(graph.getEnvironment(),
                                                   true);

            // RandomInit the Programs
            Mutator::ProgramMutator::initRandomProgram(*p, params, rng);

            // Create the action edge
            graph.addNewActionEdge(*teams.at(i), p, j);
        }

    }


    if(params.tpg.useMultiActionProgram){
        graph.updateAllAssessedActions();
    }

   
}


/*
void Mutator::TPGMutator::removeRandomActionEdge(TPG::TPGGraph& graph,
                                        const TPG::TPGVertex& vertex,
                                        Mutator::RNG& rng)
{
    // Pick an outgoing edge randomly,
    const std::list<TPG::TPGEdge*>& pickableEdges = vertex.getOutgoingActionEdges();

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
    TPG::TPGGraph& graph, const TPG::TPGVertex& vertex,
    const std::list<const TPG::TPGEdge*>& preExistingEdges, Mutator::RNG& rng)
{
    // Pick an edge (excluding ones from the team and edges with the team as a
    // destination)
    auto pickableEdges(preExistingEdges);
    // cf erase-remove idiom
    pickableEdges.erase(
        std::remove_if(
            pickableEdges.begin(), pickableEdges.end(),
            [&vertex](const TPG::TPGEdge* edge) -> bool {
                if(dynamic_cast<const TPG::TPGActionEdge*>(edge) != nullptr &&
                vertex.getAssessedActions().find(dynamic_cast<const TPG::TPGActionEdge*>(edge)->getActionClass()) ==
                vertex.getAssessedActions().end())
                {
                    return edge->getSource() == &vertex;
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
    graph.setEdgeSource(newEdge, vertex);
}




void Mutator::TPGMutator::swapActionEdges(            
    TPG::TPGGraph& graph, const TPG::TPGVertex& vertex, Mutator::RNG& rng)
{

    // Randomly select two edges
    size_t index1 = rng.getUnsignedInt64(0, vertex.getOutgoingActionEdges().size() - 1);
    size_t index2 = rng.getUnsignedInt64(0, vertex.getOutgoingActionEdges().size() - 2);
    if(index1 == index2){
        index2++;
    }

    // Use a single iterator to traverse and identify both edges
    TPG::TPGEdge* edge1 = nullptr;
    TPG::TPGEdge* edge2 = nullptr;
    size_t currentIndex = 0;

    for (auto it = vertex.getOutgoingActionEdges().begin(); it != vertex.getOutgoingActionEdges().end(); ++it, ++currentIndex) {
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
 
 */
 
 void Mutator::TPGMutator::mutateTPGActionEdge(
     TPG::TPGGraph& graph, const TPG::TPGVertex& vertex, TPG::TPGActionEdge* actionEdge,
     std::list<std::shared_ptr<Program::Program>>& newPrograms,
     const Mutator::MutationParameters& params, Mutator::RNG& rng)
{

    // copy program
    std::shared_ptr<Program::Program> newProg(
        new Program::Program(*actionEdge->getProgramSharedPointer(), true));

    // Add it to the list of new Program to be mutated.
    newPrograms.push_back(newProg);

    // Set the mutated program to the edge
    actionEdge->setProgram(newProg);
}

void Mutator::TPGMutator::duplicateEdgeSpecies(TPG::TPGGraph& graph, 
    std::vector<const TPG::TPGVertex*> species,
    std::list<std::shared_ptr<Program::Program>>& newPrograms,
    const Mutator::MutationParameters& params, Mutator::RNG& rng)
{
    
}
void Mutator::TPGMutator::deleteEdgeSpecies(TPG::TPGGraph& graph, 
    std::vector<const TPG::TPGVertex*> species,
    std::list<std::shared_ptr<Program::Program>>& newPrograms,
    const Mutator::MutationParameters& params, Mutator::RNG& rng)
{
    
}
void Mutator::TPGMutator::changeActionClassSpecies(TPG::TPGGraph& graph, 
    std::vector<const TPG::TPGVertex*> species,
    std::list<std::shared_ptr<Program::Program>>& newPrograms,
    const Mutator::MutationParameters& params, Mutator::RNG& rng)
{
    
}
void Mutator::TPGMutator::extendSpecies(TPG::TPGGraph& graph, 
    std::vector<const TPG::TPGVertex*> species,
    std::list<std::shared_ptr<Program::Program>>& newPrograms,
    const Mutator::MutationParameters& params, Mutator::RNG& rng)
{


    if(dynamic_cast<const TPG::TPGActivationVertex*>(species.front()) == nullptr){
        throw std::runtime_error("A root should always be an activation vertex");
    }
    std::cout<<1<<std::endl;

    // All individual of a species should have the exact same structure.
    // Get the first individual to know the shape of the agent
    const TPG::TPGActivationVertex* vertexExemple = (const TPG::TPGActivationVertex*)species.front();    
    std::map<const TPG::TPGDecisionVertex*, std::vector<uint64_t>> mapDecisionVertex;
    std::map<const TPG::TPGActivationVertex*, std::vector<uint64_t>> mapActivationVertex;
    if(vertexExemple->getOutgoingActionEdges().size() > 0){
        mapActivationVertex.insert(std::make_pair(vertexExemple, std::vector<size_t>()));
    }
    std::cout<<2<<std::endl;

    auto connectionEdges = vertexExemple->getOutgoingConnectionEdges();
    uint64_t indexDestinationVertex = 0;
    for(auto edge: connectionEdges){
        if(dynamic_cast<const TPG::TPGDecisionVertex*>(edge->getDestination()) != nullptr){
            mapDecisionVertex.insert(std::make_pair((const TPG::TPGDecisionVertex*)edge->getDestination(), std::vector<size_t>({indexDestinationVertex++})));
        } else {
            throw std::runtime_error("Destination of a connection edge should always be a destination vertex");
        }
    }

    std::cout<<3<<std::endl;
    // While the map of decision vertex is not empty
    while(mapDecisionVertex.size() != 0){
        auto currentPair = *mapDecisionVertex.begin();
        // Erase the first pair of the map.
        mapDecisionVertex.erase(mapDecisionVertex.begin());

        // Get the current decision vertex
        const TPG::TPGDecisionVertex* currentDecVertex = currentPair.first;

        std::cout<<"Dec "<<currentDecVertex<<std::endl;

        std::cout<<4<<std::endl;
        // For each decision edge
        uint64_t indexActivationVertex = 0;
        for(auto decisionEdge: currentDecVertex->getOutgoingEdges()){
            if(dynamic_cast<const TPG::TPGActivationVertex*>(decisionEdge->getDestination()) != nullptr){

                std::cout<<5<<std::endl;
                // Get the current activation vertex and its path
                const TPG::TPGActivationVertex* currentActVertex = (const TPG::TPGActivationVertex*)decisionEdge->getDestination();
                std::vector<uint64_t> path = currentPair.second;

                std::cout<<"Act "<<currentActVertex<<std::endl;

                // Increment the path value and update the map of activation vertex
                path.push_back(indexActivationVertex++);
                if(currentActVertex->getOutgoingActionEdges().size() > 0){
                    mapActivationVertex.insert(std::make_pair(currentActVertex, path));
                }

                // For each connection edge in the current activation vertex
                indexDestinationVertex = 0;
                for(auto connectionEdge: currentActVertex->getOutgoingConnectionEdges()){
                    if(dynamic_cast<const TPG::TPGDecisionVertex*>(connectionEdge->getDestination()) != nullptr){
                        // Get the decision vertex and copy the path
                        const TPG::TPGDecisionVertex* dest = (const TPG::TPGDecisionVertex*)connectionEdge->getDestination();
                        std::vector<uint64_t> pathDest = path;

                        std::cout<<"new dec "<<dest<<std::endl;

                        std::cout<<6<<std::endl;
                        // Update the copied path and the map of decision vertex
                        pathDest.push_back(indexDestinationVertex++);
                        mapDecisionVertex.insert(std::make_pair(dest, pathDest));

                    } else {
                        throw std::runtime_error("Destination of a connection edge should always be a destination vertex");
                    }
                }
            } else {
                throw std::runtime_error("Destination of a decision edge should always be an activation vertex");
            }
        }

        std::cout<<7<<std::endl;

        std::cout<<" "<<mapDecisionVertex.size()<<std::endl;
    }
    std::cout<<8<<std::endl;

    // Randomly select an activation vertex to do the extension on
    auto it = mapActivationVertex.begin();
    std::advance(it, rng.getUnsignedInt64(0, mapActivationVertex.size() - 1));
    std::pair<const TPG::TPGActivationVertex *const, std::vector<size_t>> selectedPair = *it;

    std::cout<<900<<std::endl;
    vertexExemple = selectedPair.first;
    std::cout<<912<<std::endl;
    std::vector<uint64_t> path = selectedPair.second;
    std::cout<<913<<std::endl;

    double probaExtendActionEdge = 0.7;
    double proba = 1;
    std::vector<uint64_t> indexActionEdges;
    uint64_t index;
    std::cout<<914<<std::endl;
    std::list<TPG::TPGEdge *> actionEdges = vertexExemple->getOrderedActionEdges();
    std::cout<<915<<" "<<actionEdges.size()<<std::endl;
    while(indexActionEdges.size() < actionEdges.size() 
          && proba > rng.getDouble(0, 1)){
        
    std::cout<<91<<std::endl;
        do {
            index = rng.getUnsignedInt64(0, actionEdges.size()-1);
        } while(std::find(indexActionEdges.begin(), indexActionEdges.end(), index) != indexActionEdges.end()) ;

        std::cout<<90<<std::endl;
        // Save the index
        indexActionEdges.push_back(index);
        proba *= probaExtendActionEdge;
    }

    std::cout<<915<<std::endl;
    // Sort the indexes for complexity later.
    std::sort(indexActionEdges.begin(), indexActionEdges.end());

    for(auto rootVertex: species){
        if(dynamic_cast<const TPG::TPGActivationVertex*>(rootVertex) == nullptr){
            throw std::runtime_error("A root should always be an activation vertex");
        }
        auto currentVertex = rootVertex;
        uint64_t indexGraph = 0;
        while(indexGraph != path.size()){

            std::list<TPG::TPGEdge *> edges;
            if(dynamic_cast<const TPG::TPGActivationVertex*>(currentVertex) != nullptr){
                edges = ((TPG::TPGActivationVertex*)currentVertex)->getOutgoingConnectionEdges();
            } else {
                edges = currentVertex->getOutgoingEdges();
            }

            // Get the new destination
            auto iterEdges = edges.begin();
            std::advance(iterEdges, path.at(indexGraph));
            currentVertex = (*iterEdges)->getDestination();

            indexGraph++;
        }

        // Create two context program
        std::shared_ptr<Program::Program> program1 = std::make_shared<Program::Program>(graph.getEnvironment(), false);
        std::shared_ptr<Program::Program> program2 = std::make_shared<Program::Program>(graph.getEnvironment(), false);
        Mutator::ProgramMutator::initRandomProgram(*program1, params, rng);
        Mutator::ProgramMutator::initRandomProgram(*program2, params, rng);

        // Create two activation vertex and a decision vertex
        const TPG::TPGActivationVertex& actVertex1 = graph.addNewActivationVertex();
        const TPG::TPGActivationVertex& actVertex2 = graph.addNewActivationVertex();
        const TPG::TPGDecisionVertex& decVertex = graph.addNewDecisionVertex();

        // Add a connexion edge between the current vertex and the decision vertex
        graph.addNewConnectionEdge(*currentVertex, decVertex);

        // Connect each context program to one team
        graph.addNewDecisionEdge(decVertex, actVertex1, program1);
        graph.addNewDecisionEdge(decVertex, actVertex2, program2);

        // Get the action edges from the indexes in indexActionEdges 
        std::list<TPG::TPGEdge *> listEdgesExtended;

        if(dynamic_cast<const TPG::TPGActivationVertex*>(currentVertex) == nullptr){
            throw std::runtime_error("The current vertex should be an activation vertex");
        }

        auto actionEdges = ((const TPG::TPGActivationVertex*)currentVertex)->getOrderedActionEdges();
        auto iterActionEdges = actionEdges.begin();
        std::advance(iterActionEdges, indexActionEdges[0]);
        listEdgesExtended.push_back(*iterActionEdges);


        size_t currentIndex = 0;
        while(listEdgesExtended.size() < indexActionEdges.size()){
            std::advance(iterActionEdges, indexActionEdges[currentIndex+1] - indexActionEdges[currentIndex]);
            listEdgesExtended.push_back(*iterActionEdges);
            currentIndex++;
        }

        for(auto actionEdge: listEdgesExtended){
            uint64_t actionClass = ((TPG::TPGActionEdge*)actionEdge)->getActionClass();

            // For one team, just add an edge with the same shared_ptr
            graph.addNewActionEdge(actVertex1, actionEdge->getProgramSharedPointer(), actionClass);

            // For the other team, duplicate the program and add it to the newPrograms list
            std::shared_ptr<Program::Program> newProg(new Program::Program(*actionEdge->getProgramSharedPointer(), true));
            newPrograms.push_back(newProg);
            graph.addNewActionEdge(actVertex2, newProg, actionClass);

            // remove the action edges duplicated from the original team
            graph.removeEdge(*actionEdge);
        }


    }
    


}

void Mutator::TPGMutator::mutateSpecies(TPG::TPGGraph& graph, 
    std::vector<const TPG::TPGVertex*> species,
    std::list<std::shared_ptr<Program::Program>>& newPrograms,
    const Mutator::MutationParameters& params, Mutator::RNG& rng)
{
    double probaDuplicationEdge = 0.0;
    double probaDeletionEdge = 0.0;
    double probaChangeActionClass = 0.0;
    double probaExtension = 1.0;

    double mutationValue = rng.getDouble(0.0, probaDuplicationEdge + probaDeletionEdge + probaChangeActionClass + probaExtension);

    if(probaDuplicationEdge > mutationValue){
        duplicateEdgeSpecies(graph, species, newPrograms, params, rng);
    } else if(probaDuplicationEdge + probaDeletionEdge > mutationValue){
        deleteEdgeSpecies(graph, species, newPrograms, params, rng);
    } else if(probaDuplicationEdge + probaDeletionEdge + probaChangeActionClass > mutationValue){
        changeActionClassSpecies(graph, species, newPrograms, params, rng);
    } else {
        extendSpecies(graph, species, newPrograms, params, rng);
    }
}

void Mutator::TPGMutator::mutateTPGVertex(
    TPG::TPGGraph& graph, const TPG::TPGVertex& vertex,
    std::list<std::shared_ptr<Program::Program>>& newPrograms,
    const Mutator::MutationParameters& params, Mutator::RNG& rng)
{
    std::cout<<"change this"<<std::endl;
    return;


    auto outgoingActionEdges = ((TPG::TPGActivationVertex*) &vertex)->getOrderedActionEdges();


    bool anyMutationDone = false;
    do {
        std::vector<uint64_t> indexUsed;
        uint64_t index;
        // 4. mutate randomly selected program on action Edge. 
        double proba = params.tpg.pMutateActionProgram;
        while(indexUsed.size() < outgoingActionEdges.size()  && proba > rng.getDouble(0.0, 1.0)){

            do {
                index = rng.getUnsignedInt64(0, outgoingActionEdges.size()-1);
            } while(std::find(indexUsed.begin(), indexUsed.end(), index) != indexUsed.end()) ;

            indexUsed.push_back(index);
    
            std::list<TPG::TPGEdge *>::const_iterator iter = outgoingActionEdges.begin();
            std::advance(iter, index);
            TPG::TPGActionEdge* actionEdge = dynamic_cast<TPG::TPGActionEdge*>(*iter);

            mutateTPGActionEdge(graph, vertex, actionEdge, newPrograms, params, rng);

            proba *= params.tpg.pMutateActionProgram;

            anyMutationDone = true;
        }
    } while (!anyMutationDone && params.tpg.pMutateActionProgram != 0.0);


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
        newProgCopy = std::make_shared<Program::Program>(
            *newProg, newProg->isActionProgram());
    }

    bool allUnique;
    // Mutate behavior until it changes (against the archive).
    do {

        // If a new program is created
        if (rng.getDouble(0.0, 1.0) < params.prog.pNewProgram) {
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

        // Do not use Archive right now if the environment is continuous
        // TODO Update that
    } while (!allUnique &&
             (newProg->getEnvironment().getNbContinuousActions() == 0 || 
             params.tpg.useMultiActionProgram));
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

void Mutator::TPGMutator::crossProgram(
    TPG::TPGGraph& graph,
    std::vector<const TPG::TPGActivationVertex*> childs,
    size_t actionID,
    const Mutator::MutationParameters& params,
    Mutator::RNG& rng)
{

    // Create new empty programs
    std::array<std::shared_ptr<Program::Program>, 2> newProgs = {
        std::make_shared<Program::Program>(graph.getEnvironment(), true),
        std::make_shared<Program::Program>(graph.getEnvironment(), true)
    };

    // Get the programs of the parents, it should alreay be checked that program exist.
    std::array<std::shared_ptr<Program::Program>, 2> originProgs = {
        childs.at(0)->getEdgeOfAction(actionID)->getProgramSharedPointer(),
        childs.at(1)->getEdgeOfAction(actionID)->getProgramSharedPointer()
    };

    std::array<uint64_t, 2> cutStart, cutEnd, sizeProgs;

    // if the sum of the parents program size is above the max size, the size of the cross lines is the same for both parents.
    bool specialCase = originProgs[0]->getNbLines() + originProgs[1]->getNbLines() >= params.prog.maxProgramSize;

    // Select random index for the crossover, normal case
    for (int i = 0; i < 2; i++) {

        uint64_t nbLines = originProgs[i]->getNbLines();
        if(specialCase){
            nbLines = std::min(originProgs[0]->getNbLines(), originProgs[1]->getNbLines());
        }

        if (nbLines < 2) return; // If a program has only one line, crossover cannot happen.

        cutStart[i] = rng.getUnsignedInt64(0, nbLines - 1);
        cutEnd[i] = rng.getUnsignedInt64(0, nbLines - 2);
        if (cutEnd[i] == cutStart[i]) {
            cutEnd[i]++;
        } else if (cutEnd[i] < cutStart[i]) {
            std::swap(cutStart[i], cutEnd[i]);
        }

        if(specialCase){
            cutStart[1] = cutStart[0];
            cutEnd[1] = cutEnd[0];
            break;
        }
    }



    // Compute program size of the children
    for (int i = 0; i < 2; i++) {
        sizeProgs[i] = originProgs[i]->getNbLines() - (cutEnd[i] - cutStart[i]) + (cutEnd[1 - i] - cutStart[1 - i]);
    }

    // Create new programs with the cut
    for (int childIdx = 0; childIdx < 2; childIdx++) {
        auto& newProg = newProgs[childIdx];
        auto& parent1 = originProgs[childIdx];
        auto& parent2 = originProgs[1 - childIdx];
        uint64_t start1 = cutStart[childIdx], end1 = cutEnd[childIdx];
        uint64_t start2 = cutStart[1 - childIdx], end2 = cutEnd[1 - childIdx];

        for (size_t idx = 0; idx < sizeProgs[childIdx]; idx++) {
            if (idx < start1) {
                newProg->addNewLine(parent1->getLine(idx));
            } else if (idx >= start1 + (end2 - start2)) {
                newProg->addNewLine(parent1->getLine(idx + (end1 - start1) - (end2 - start2)));
            } else {
                newProg->addNewLine(parent2->getLine(idx - start1 + start2));
            }
        }
    }

    // Add the new programs to the child.
    for (int i = 0; i < 2; i++) {
        graph.addNewActionEdge(*childs.at(i), newProgs[i], actionID);
        graph.removeEdge(*childs.at(i)->getEdgeOfAction(actionID));
        newProgs[i]->identifyIntrons();
    }

}

void Mutator::TPGMutator::crossEdges(
    TPG::TPGGraph& graph,
    std::vector<const TPG::TPGActivationVertex*> childs,
    size_t actionID,
    const Mutator::MutationParameters& params,
    Mutator::RNG& rng)
{


    // get the edges
    TPG::TPGActionEdge* edge1 = childs.at(0)->getEdgeOfAction(actionID);
    TPG::TPGActionEdge* edge2 = childs.at(1)->getEdgeOfAction(actionID);

    // Only add the edge if the action is founded.
    if(edge1 != nullptr){
        graph.addNewActionEdge(*childs.at(1), edge1->getProgramSharedPointer(), actionID);
        graph.removeEdge(*edge1);
    }
    
    // Only add the edge if the action is founded.
    if(edge2 != nullptr){
        graph.addNewActionEdge(*childs.at(0), edge2->getProgramSharedPointer(), actionID);
        graph.removeEdge(*edge2);
    }
}

void Mutator::TPGMutator::crossTPGVertices(
    TPG::TPGGraph& graph,
    std::vector<const TPG::TPGActivationVertex*> childs,
    const Mutator::MutationParameters& params,
    Mutator::RNG& rng)
{

    std::vector<uint64_t> indexUsed;
    uint64_t indexAction;
    
    // Always do at least one crossover, except is the proba is at zero (mearning we don't want any crossover)
    double proba = (params.tpg.probaCrossAgents != 0) ? 1: 0;
    while(indexUsed.size() < graph.getEnvironment().getNbContinuousActions()  && proba > rng.getDouble(0.0, 1.0)){


        // Select the action ID
        do {
            indexAction = rng.getUnsignedInt64(0, graph.getEnvironment().getNbContinuousActions()-1);
        } while(std::find(indexUsed.begin(), indexUsed.end(), indexAction) != indexUsed.end()) ;

        indexUsed.push_back(indexAction);

        // A crossover at program level can be done only the both parents assessed the action concerned
        if(childs.at(0)->getAssessedActions().count(indexAction) > 0 &&
        childs.at(1)->getAssessedActions().count(indexAction) > 0 &&
        params.tpg.probaCrossPrograms > rng.getDouble(0, 1)){
            
            if(params.tpg.typeProgramCrossover == "standard"){
                crossProgram(graph, childs, indexAction, params, rng);
            } else {
                throw std::runtime_error("params.mutation.tpg.typeProgramCrossover not found");
            }



        } else {
            crossEdges(graph, childs, indexAction, params, rng);
        }
        proba *= params.tpg.probaCrossAgents;
    }

    for(auto child: childs){
        graph.updateAssessedActions(child);
    }


}

void Mutator::TPGMutator::populateTPG(TPG::TPGGraph& graph,
                                      const Archive& archive,
                                      const Mutator::MutationParameters& params,
                                      Mutator::RNG& rng, uint64_t nbActions,
                                      uint64_t maxNbThreads)
{
    std::cout<<"ehre"<<std::endl;
    // Get current vertex set (copy)
    auto vertices(graph.getVertices());
    // Get current root teams (copy)
    auto rootVertices(graph.getRootVertices());





    // Create an empty list to store Programs to mutate.
    std::list<std::shared_ptr<Program::Program>> newPrograms;



    bool useTournamentSelection = graph.getEnvironment().getParams().useTournamentSelection;
    if (useTournamentSelection) {
        // The root not set to be deleted are not used during evolution
        rootVertices.erase(
            std::remove_if(rootVertices.begin(), rootVertices.end(),
                           [](const TPG::TPGVertex* vertex) -> bool {
                               return !vertex->isToBeDeleted();}),
                               rootVertices.end());
    }

    std::cout<<"ehre2"<<std::endl;

    uint64_t nbRootsToCreate = params.tpg.nbRoots - graph.getNbRootVertices() + (rootVertices.size() * useTournamentSelection);

    std::vector<const TPG::TPGVertex*> rootUsedParents1 = rootVertices;
    std::vector<const TPG::TPGVertex*> rootUsedParents2;
    if(useTournamentSelection){
        // Divide root used into two subVector with half of the roots, randomly selected.
        for(size_t idx = 0; idx < rootVertices.size() / 2; idx++){
            auto root = rootUsedParents1.at(rng.getUnsignedInt64(0, rootUsedParents1.size() - 1));
    
            rootUsedParents2.push_back(root);
            std::swap(root, rootUsedParents1.back());
            rootUsedParents1.pop_back();
        }
    } else {
        rootUsedParents2 = rootVertices;
    }

    std::cout<<"ehre2"<<std::endl;

    uint64_t nbRootsCreated = 0;
    while (nbRootsCreated < nbRootsToCreate) {

        // Not really clean but efficient switching between tournament and not tournament selection
        // Select a random existing root
        uint64_t clonedRootIndex1 =
            rng.getUnsignedInt64(0, rootUsedParents1.size() - 1);
        // Select a random existing root
        uint64_t clonedRootIndex2 =
            rng.getUnsignedInt64(0, rootUsedParents2.size() - 2 + useTournamentSelection);
        
        // Be sure it is different
        if(clonedRootIndex1 == clonedRootIndex2 && !useTournamentSelection){
            clonedRootIndex2++;
        }

        std::cout<<"ehre2"<<std::endl;

        std::cout<<"ehre2321"<<std::endl;
        const TPG::TPGActivationVertex* child1 = (const TPG::TPGActivationVertex*)&graph.cloneVertex(*rootUsedParents1.at(clonedRootIndex1));
        const TPG::TPGActivationVertex* child2 = (const TPG::TPGActivationVertex*)&graph.cloneVertex(*rootUsedParents2.at(clonedRootIndex2));

        std::cout<<"ehre2321"<<std::endl;
        // Get parents and create childs
        std::vector<const TPG::TPGActivationVertex*> childs{child1, child2};

        std::cout<<"ehre2322"<<std::endl;
        // Do the crossover over the childs
        //crossTPGVertices(graph, childs, params, rng);

        // Do the mutation over the childs
        for(auto child: childs){
            if(child->getOutgoingEdges().size() == 0){
                graph.removeVertex(*child);
                nbRootsCreated--;
            } else {
                std::cout<<"ehre231"<<std::endl;
                mutateTPGVertex(graph, *child, newPrograms,
                                params, rng);
            }
        }

        // Check the new number of roots
        // Needed since preExisting root may be subsumed by new ones.
        nbRootsCreated += 2;
    }

    bool v = false;
    for(auto root: rootVertices){
        if(root->isToBeDeleted()){
            graph.removeVertex(*root);
            v = true;
        }
    }
    double probaMutateSpecies = 1.0;
    if(probaMutateSpecies > rng.getDouble(0, 1) && v){
        std::cout<<"ehre2"<<std::endl;
        mutateSpecies(graph, graph.getRootVertices(), newPrograms, params, rng);
        std::cout<<"ehre2"<<std::endl;
    }

    // Mutate the new Programs
    mutateNewProgramBehaviors(maxNbThreads, newPrograms, rng, params, archive);
    std::cout<<"ehre"<<std::endl;
}
