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
        teams.push_back(&(graph.addNewActivationVertex({0})));

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
        graph.orderOutgoingEdges(teams.back());

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
 
 void Mutator::TPGMutator::mutateTPGEdge(
     TPG::TPGGraph& graph, const TPG::TPGVertex& vertex, TPG::TPGEdge* edge,
     std::list<std::shared_ptr<Program::Program>>& newPrograms,
     const Mutator::MutationParameters& params, Mutator::RNG& rng)
{

    // copy program
    std::shared_ptr<Program::Program> newProg(
        new Program::Program(*edge->getProgramSharedPointer(), true));

    // Add it to the list of new Program to be mutated.
    newPrograms.push_back(newProg);

    // Set the mutated program to the edge
    edge->setProgram(newProg);
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

    // All individual of a species should have the exact same structure.
    // Get the first individual to know the shape of the agent
    std::vector<const TPG::TPGVertex *> verticesOfExemple = graph.getVerticesOfRoot((const TPG::TPGActivationVertex*)species.front());

    std::vector<const TPG::TPGVertex *> weightedVerticesOfExemple;

    for (const auto* vertex : verticesOfExemple) {

        // If Vertex is activation vertex, add i N times, with N is the number of action in the vertex
        if (auto activationVertex = dynamic_cast<const TPG::TPGActivationVertex*>(vertex)) {
            weightedVerticesOfExemple.insert(weightedVerticesOfExemple.end(), std::pow(activationVertex->getOutgoingActionEdges().size(),2), vertex);
        }
    }

    // Randomly select an activation vertex to do the extension on based on the weight
    uint64_t indexChoosenWeightedVertex = rng.getUnsignedInt64(0, weightedVerticesOfExemple.size() - 1);

    // Fing the corresponding index in the original vector.
    auto it = std::find(verticesOfExemple.begin(), verticesOfExemple.end(), weightedVerticesOfExemple.at(indexChoosenWeightedVertex));
    uint64_t indexChoosenVertex = std::distance(verticesOfExemple.begin(), it);

    // Get the vertex and the path
    const TPG::TPGActivationVertex* vertexExemple = (const TPG::TPGActivationVertex*)verticesOfExemple.at(indexChoosenVertex);
    std::vector<uint64_t> path = vertexExemple->getPath();


    double probaExtendActionEdge = 0.9;
    double proba = 1;
    std::vector<uint64_t> indexActionEdges;
    uint64_t index;
    std::list<TPG::TPGEdge *> actionEdgesExemple = vertexExemple->getOutgoingActionEdges();
    while(indexActionEdges.size() < actionEdgesExemple.size() 
          && proba > rng.getDouble(0, 1)){
        
        do {
            index = rng.getUnsignedInt64(0, actionEdgesExemple.size()-1);
        } while(std::find(indexActionEdges.begin(), indexActionEdges.end(), index) != indexActionEdges.end()) ;

        // Save the index
        indexActionEdges.push_back(index);
        proba *= probaExtendActionEdge;
    }

    // Sort the indexes for complexity later.
    std::sort(indexActionEdges.begin(), indexActionEdges.end());

    for(auto rootVertex: species){
        if(dynamic_cast<const TPG::TPGActivationVertex*>(rootVertex) == nullptr){
            throw std::runtime_error("A root should always be an activation vertex");
        }
        auto currentVertex = graph.getVerticesOfRoot(rootVertex).at(indexChoosenVertex);
        uint64_t indexGraph = 0;
        

        // Create two context program
        std::shared_ptr<Program::Program> program1 = std::make_shared<Program::Program>(graph.getEnvironment(), false);
        std::shared_ptr<Program::Program> program2 = std::make_shared<Program::Program>(graph.getEnvironment(), false);
        Mutator::ProgramMutator::initRandomProgram(*program1, params, rng);
        Mutator::ProgramMutator::initRandomProgram(*program2, params, rng);
        std::vector<uint64_t> decPath = currentVertex->getPath();
        decPath.push_back(currentVertex->getOutgoingEdges().size() - actionEdgesExemple.size());
        const TPG::TPGDecisionVertex& decVertex = graph.addNewDecisionVertex(decPath);

        // Create two activation vertex and a decision vertex
        decPath.push_back(0);
        const TPG::TPGActivationVertex& actVertex1 = graph.addNewActivationVertex(decPath);
        decPath.back() = 1;
        const TPG::TPGActivationVertex& actVertex2 = graph.addNewActivationVertex(decPath);

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

        auto actionEdges = ((const TPG::TPGActivationVertex*)currentVertex)->getOutgoingActionEdges();
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

    for(auto vertex: species){
        graph.orderOutgoingEdges(vertex);
    }
}

void Mutator::TPGMutator::mutateTPGVertex(
    TPG::TPGGraph& graph, const TPG::TPGVertex& vertex,
    std::list<std::shared_ptr<Program::Program>>& newPrograms,
    const Mutator::MutationParameters& params, Mutator::RNG& rng)
{

    std::vector<TPG::TPGEdge*> allEdges = graph.getEdgesOfRoot(&vertex, false);


    bool anyMutationDone = false;
    do {
        std::vector<uint64_t> indexUsed;
        uint64_t index;
        // 4. mutate randomly selected program on action Edge. 
        double proba = params.tpg.pMutateActionProgram;
        while(indexUsed.size() < allEdges.size()  && proba > rng.getDouble(0.0, 1.0)){

            do {
                index = rng.getUnsignedInt64(0, allEdges.size()-1);
            } while(std::find(indexUsed.begin(), indexUsed.end(), index) != indexUsed.end()) ;

            indexUsed.push_back(index);
    
            auto iter = allEdges.begin();
            std::advance(iter, index);
            TPG::TPGActionEdge* actionEdge = dynamic_cast<TPG::TPGActionEdge*>(*iter);

            mutateTPGEdge(graph, vertex, *iter, newPrograms, params, rng);

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
    std::vector<const TPG::TPGVertex*>& childs,
    std::vector<TPG::TPGEdge*>& edges,
    const Mutator::MutationParameters& params,
    Mutator::RNG& rng)
{

    bool actionProgram = edges.at(0)->getProgramSharedPointer()->isActionProgram();

    // Create new empty programs
    std::array<std::shared_ptr<Program::Program>, 2> newProgs = {
        std::make_shared<Program::Program>(graph.getEnvironment(), actionProgram),
        std::make_shared<Program::Program>(graph.getEnvironment(), actionProgram)
    };

    // Get the programs of the parents, it should alreay be checked that program exist.
    std::array<std::shared_ptr<Program::Program>, 2> originProgs = {
        edges.at(0)->getProgramSharedPointer(),
        edges.at(1)->getProgramSharedPointer()
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
        edges.at(i)->setProgram(newProgs[i]);
        newProgs[i]->identifyIntrons();
    }

}

void Mutator::TPGMutator::crossEdges(
    TPG::TPGGraph& graph,
    std::vector<const TPG::TPGVertex*>& childs,
    std::vector<TPG::TPGEdge*>& edges,
    const Mutator::MutationParameters& params,
    Mutator::RNG& rng)
{
    // Create new edges depending of edge type
    for (int i = 0; i < 2; i++) {
        if (auto connEdge = dynamic_cast<TPG::TPGConnectionEdge*>(edges.at(0))) {
            graph.addNewConnectionEdge(*childs.at(1 - i), *edges.at(i)->getDestination());
        } 
        else if (auto actionEdge = dynamic_cast<TPG::TPGActionEdge*>(edges.at(0))) {
            size_t actionID = actionEdge->getActionClass();
            graph.addNewActionEdge(*childs.at(1 - i), edges.at(i)->getProgramSharedPointer(), actionID);
        } 
        else {
            graph.addNewDecisionEdge(*childs.at(1 - i), *edges.at(i)->getDestination(), edges.at(i)->getProgramSharedPointer());
        }
    }

    // Remove former edges
    for (auto edge : edges) {
        graph.removeEdge(*edge);
    }
}

void Mutator::TPGMutator::crossTPGVertices(
    TPG::TPGGraph& graph,
    std::vector<const TPG::TPGVertex*>& childs,
    const Mutator::MutationParameters& params,
    Mutator::RNG& rng)
{




    if(params.tpg.probaCrossAgents > rng.getDouble(0.0, 1.0)){

        // Select a random vertex in the graph of the child
        std::vector<const TPG::TPGVertex*> vertices = graph.getVerticesOfRoot(childs.at(0));

        size_t indexVertex = rng.getUnsignedInt64(0, vertices.size() - 1);
        const TPG::TPGVertex* selectedVertex1 = vertices.at(indexVertex);
        const TPG::TPGVertex* selectedVertex2 = graph.getVerticesOfRoot(childs.at(1)).at(indexVertex);
        std::vector<const TPG::TPGVertex*> childsCrossed = {selectedVertex1, selectedVertex2};



        size_t indexEdges = rng.getUnsignedInt64(0, selectedVertex1->getOutgoingEdges().size() - 1);

        // get the edges
        auto it1 = childsCrossed.at(0)->getOutgoingEdges().begin();
        std::advance(it1, indexEdges);
        TPG::TPGEdge* edge1 = *it1;
        
        auto it2 = childsCrossed.at(1)->getOutgoingEdges().begin();
        std::advance(it2, indexEdges);
        TPG::TPGEdge* edge2 = *it2;

        std::vector<TPG::TPGEdge*> edgesCrossed = {edge1, edge2};
        crossEdges(graph, childsCrossed, edgesCrossed, params, rng);

        // Update order and assessed actions
        for(auto child: childsCrossed){
            graph.updateAssessedActions(child);
            graph.orderOutgoingEdges(child);
        }


        // get the new edges
        it1 = childsCrossed.at(0)->getOutgoingEdges().begin();
        std::advance(it1, indexEdges);
        edge1 = *it1;
        
        it2 = childsCrossed.at(1)->getOutgoingEdges().begin();
        std::advance(it2, indexEdges);
        edge2 = *it2;

        // Get all the copied edges except connection edges
        std::vector<TPG::TPGEdge*> edgesCopied1;
        std::vector<TPG::TPGEdge*> edgesCopied2;
        if(dynamic_cast<TPG::TPGActionEdge*>(edge1) == nullptr){
            edgesCopied1 = graph.getEdgesOfRoot(edge1->getDestination(), false);
            edgesCopied2 = graph.getEdgesOfRoot(edge2->getDestination(), false);
        }
        if(dynamic_cast<TPG::TPGConnectionEdge*>(edge1) == nullptr){
            edgesCopied1.push_back(edge1);
            edgesCopied2.push_back(edge2);
        }


        std::vector<uint64_t> indicesUsed;
        uint64_t currentIndex;

        // Always do at least one crossover, except is the proba is at zero (mearning we don't want any crossover)
        double proba = (params.tpg.probaCrossPrograms != 0) ? 1: 0;
        while(indicesUsed.size() < edgesCopied1.size()  && proba > rng.getDouble(0.0, 1.0)){


            // Select the action ID
            do {
                currentIndex = rng.getUnsignedInt64(0, edgesCopied1.size()-1);
            } while(std::find(indicesUsed.begin(), indicesUsed.end(), currentIndex) != indicesUsed.end()) ;

            indicesUsed.push_back(currentIndex);

            std::vector<TPG::TPGEdge*> programEdgesCrossed = {edgesCopied1.at(currentIndex), edgesCopied2.at(currentIndex)};

            if(params.tpg.typeProgramCrossover == "standard"){
                crossProgram(graph, childsCrossed, programEdgesCrossed, params, rng);
            } else {
                throw std::runtime_error("params.mutation.tpg.typeProgramCrossover not found");
            }



            proba *= params.tpg.probaCrossPrograms;
        }
    }




    


}

void Mutator::TPGMutator::populateTPG(TPG::TPGGraph& graph,
                                      const Archive& archive,
                                      const Mutator::MutationParameters& params,
                                      Mutator::RNG& rng, uint64_t nbActions,
                                      uint64_t maxNbThreads)
{
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


        const TPG::TPGVertex* child1 = &graph.cloneVertex(*rootUsedParents1.at(clonedRootIndex1));
        const TPG::TPGVertex* child2 = &graph.cloneVertex(*rootUsedParents2.at(clonedRootIndex2));

        // Get parents and create childs
        std::vector<const TPG::TPGVertex*> childs{child1, child2};

        // Do the crossover over the childs
        crossTPGVertices(graph, childs, params, rng);

        // Do the mutation over the childs
        for(auto child: childs){
            if(child->getOutgoingEdges().size() == 0){
                graph.removeVertex(*child);
                nbRootsCreated--;
            } else {
                mutateTPGVertex(graph, *child, newPrograms,
                                params, rng);
            }
        }

        // Check the new number of roots
        // Needed since preExisting root may be subsumed by new ones.
        nbRootsCreated += 2;
    }

    for(auto root: rootVertices){
        if(root->isToBeDeleted()){
            graph.removeVertex(*root);
        }
    }
    double probaMutateSpecies = 1.0;
    if(probaMutateSpecies > rng.getDouble(0, 1) && nbActions > 0 && nbActions < 2){
        mutateSpecies(graph, graph.getRootVertices(), newPrograms, params, rng);
    }

    // Mutate the new Programs
    mutateNewProgramBehaviors(maxNbThreads, newPrograms, rng, params, archive);
}
