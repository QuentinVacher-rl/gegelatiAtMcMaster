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



    // Create only one species for now, with one activation vertex and nbActionEdgeInit actionEdge.
    const TPG::TPGActivationVertex& vertex = graph.addNewActivationVertex();
    for(size_t actionValue = 0; actionValue < nbActionEdgeInit; actionValue++){
        graph.addNewActionEdge(vertex, actionValue);
    }
    graph.addSpecies(vertex);

    // Create agents
    for(size_t indexAgent = 0; indexAgent < params.tpg.nbRoots; indexAgent++){
        
        const TPG::TPGAgent& agent = graph.addNewAgent(vertex);

        for(auto edge: vertex.getOutgoingEdges()){

            
            // Create a program and specify action program
            std::shared_ptr<Program::Program> prog =
                std::make_shared<Program::Program>(graph.getEnvironment(),
                                                   true);

            // RandomInit the Programs.
            Mutator::ProgramMutator::initRandomProgram(*prog, params, rng);

            // Add the program to the agent.
            graph.setProgramToAgent(agent, edge, prog);
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
 

bool Mutator::TPGMutator::addEdgeSpecies(TPG::TPGGraph& graph, 
    const TPG::TPGVertex* species,
    std::list<std::shared_ptr<Program::Program>>& newPrograms,
    const Mutator::MutationParameters& params, Mutator::RNG& rng)
{
    return true;
    double probaActivationOverDecisionVertex = 1;
    double randomValue = rng.getDouble(0, 1);

    std::vector<const TPG::TPGVertex *> verticesOfExemple = graph.getVerticesOfRoot(species);
    std::vector<const TPG::TPGVertex *> verticesUsed = verticesOfExemple;

    // Remove vertices with only one action for activation vertex or one edge for decision vertex
    verticesUsed.erase(
        std::remove_if(
            verticesUsed.begin(), verticesUsed.end(),
            [probaActivationOverDecisionVertex, randomValue](const TPG::TPGVertex* vertex) -> bool {
                // If proba > random, keep only activation vertices, else keep only decision vertices.
                return (probaActivationOverDecisionVertex > randomValue) ^ dynamic_cast<const TPG::TPGActivationVertex*>(vertex) != nullptr;
            }
        ),
        verticesUsed.end()
    );

    if(probaActivationOverDecisionVertex > randomValue){
        std::set<uint64_t> rootAvailableActions;

        // Get the actions assessed by the root
        const std::set<uint64_t>& actionAssessedByRoot = species->getAssessedActions();
        for (uint64_t i = 0; i < graph.getEnvironment().getNbContinuousActions(); ++i) {
            // If action not assessed by the root, it is available.
            if (actionAssessedByRoot.find(i) == actionAssessedByRoot.end()) {
                rootAvailableActions.insert(i);
            }
        }
    
        // Get all the vertex with available actions, take a vector for keeping track of the order
        std::vector<std::pair<const TPG::TPGVertex*, std::set<uint64_t>>> availableActionsAllVertices;
        for(auto vertex: verticesUsed){
    
            std::set<uint64_t> availableActionOfVertex = rootAvailableActions;
    
            const TPG::TPGActivationVertex* currentVertex = (const TPG::TPGActivationVertex*)vertex;
            while(currentVertex->getIncomingEdges().size() > 0){
                if(dynamic_cast<const TPG::TPGDecisionVertex*>(currentVertex->getIncomingEdges().front()->getSource()) == nullptr){
                    throw std::runtime_error("Should be a decision vertex (addEdgeSpecies)");
                }
        
                // Get the incomming decision vertex
                const TPG::TPGVertex* decisionVertex = currentVertex->getIncomingEdges().front()->getSource();
        
                auto avSet = currentVertex->getAssessedActions();
                auto dvSet = decisionVertex->getAssessedActions();
    
                if(dvSet.size() < avSet.size()){
                    throw std::runtime_error("Size of the decision vertex assessed actions should be equal or bigger (addEdgeSpecies)");
                }
        
                // Add the set difference in the available action set.
                std::set_difference(
                    dvSet.begin(), dvSet.end(),
                    avSet.begin(), avSet.end(),
                    std::inserter(availableActionOfVertex, availableActionOfVertex.begin())
                );
    
    
                
        
                if(dynamic_cast<const TPG::TPGActivationVertex*>(decisionVertex->getIncomingEdges().front()->getSource()) == nullptr){
                    throw std::runtime_error("Should be an action vertex (addEdgeSpecies)");
                }
        
                // Get back to thr root
                currentVertex = (const TPG::TPGActivationVertex*)decisionVertex->getIncomingEdges().front()->getSource();
            }
        
            if(availableActionOfVertex.size() > 0){
                availableActionsAllVertices.push_back(std::make_pair(vertex, availableActionOfVertex));
            }
        }
    
        if(availableActionsAllVertices.size() == 0){
            return false;
        }
    
        // Select randomly the vertex.
        auto itMap = availableActionsAllVertices.begin();
        std::advance(itMap, rng.getUnsignedInt64(0, availableActionsAllVertices.size() - 1));
        auto pair = *itMap;

        // Fing the corresponding index in the original vector.
        auto itVertex = std::find(verticesOfExemple.begin(), verticesOfExemple.end(), (*itMap).first);
        uint64_t indexChoosenVertex = std::distance(verticesOfExemple.begin(), itVertex);

        // Randomly select an action
        auto actionSetChoosen = (*itMap).second;
        auto itAction = actionSetChoosen.begin();
        std::advance(itAction, rng.getUnsignedInt64(0, actionSetChoosen.size() - 1));
        uint64_t actionChoosen = (*itAction);
    
        /*for(auto rootVertex: species){
            if(dynamic_cast<const TPG::TPGActivationVertex*>(rootVertex) == nullptr){
                throw std::runtime_error("A root should always be an activation vertex");
            }
            auto currentVertex = graph.getVerticesOfRoot(rootVertex).at(indexChoosenVertex);
    
            // Create a random program
            std::shared_ptr<Program::Program> program = std::make_shared<Program::Program>(graph.getEnvironment(), true);
            Mutator::ProgramMutator::initRandomProgram(*program, params, rng);
            //graph.addNewActionEdge(*currentVertex, program, actionChoosen);
        }*/
    } else {
    
        if(verticesUsed.size() == 0){
            return false;
        }
        // Select randomly the vertex.
        auto itUsed = verticesUsed.begin();
        std::advance(itUsed, rng.getUnsignedInt64(0, verticesUsed.size() - 1));
        const TPG::TPGVertex* vertexChoosen = *itUsed;

        // Fing the corresponding index in the original vector.
        auto itVertex = std::find(verticesOfExemple.begin(), verticesOfExemple.end(), vertexChoosen);
        uint64_t indexChoosenVertex = std::distance(verticesOfExemple.begin(), itVertex);
        // Randomly select an edge, whose vertex will be duplicated.
        uint64_t indexEdgeChoosen = rng.getUnsignedInt64(0, vertexChoosen->getOutgoingEdges().size() - 1);
    
        /*for(auto rootVertex: species){
            if(dynamic_cast<const TPG::TPGActivationVertex*>(rootVertex) == nullptr){
                throw std::runtime_error("A root should always be an activation vertex");
            }
            auto currentVertex = graph.getVerticesOfRoot(rootVertex).at(indexChoosenVertex);

            if(dynamic_cast<const TPG::TPGDecisionVertex*>(currentVertex) == nullptr){
                throw std::runtime_error("The vertex here should always be a decision vertex");
            }

            auto itEdge = currentVertex->getOutgoingEdges().begin();
            std::advance(itEdge, indexEdgeChoosen);
            const TPG::TPGVertex* vertexDestination = &graph.cloneVertex(*(*itEdge)->getDestination());


            // Create a random program
            std::shared_ptr<Program::Program> program = std::make_shared<Program::Program>(graph.getEnvironment(), false);
            Mutator::ProgramMutator::initRandomProgram(*program, params, rng);
            //graph.addNewDecisionEdge(*currentVertex, *vertexDestination, program);
        }*/
    }
    

    return true;
    
}

bool Mutator::TPGMutator::deleteEdgeSpecies(TPG::TPGGraph& graph, 
    const TPG::TPGVertex* species,
    std::list<std::shared_ptr<Program::Program>>& newPrograms,
    const Mutator::MutationParameters& params, Mutator::RNG& rng)
{
    return true;
    std::vector<const TPG::TPGVertex *> verticesOfExemple = graph.getVerticesOfRoot((const TPG::TPGActivationVertex*)species);
    std::vector<const TPG::TPGVertex *> verticesUsed = verticesOfExemple;

    // Remove activation vertices with only one edge and decision vertices with two edges or less.
    verticesUsed.erase(
        std::remove_if(
            verticesUsed.begin(), verticesUsed.end(),
            [](const TPG::TPGVertex* vertex) -> bool {
                if(dynamic_cast<const TPG::TPGActivationVertex*>(vertex) != nullptr){
                    return vertex->getOutgoingEdges().size() < 2;
                } else {
                    return vertex->getOutgoingEdges().size() < 3;
                }
            }
        ),
        verticesUsed.end()
    );

    if(verticesUsed.size() == 0){
        return false;
    }

    // Select randomly the vertex.
    auto it = verticesUsed.begin();
    std::advance(it, rng.getUnsignedInt64(0, verticesUsed.size() - 1));
    auto vertex = (const TPG::TPGActivationVertex*)*it;

    // Find the corresponding index in the original vector.
    auto itVertex = std::find(verticesOfExemple.begin(), verticesOfExemple.end(), (*it));
    uint64_t indexChoosenVertex = std::distance(verticesOfExemple.begin(), itVertex);

    auto edgeIndex = rng.getUnsignedInt64(0, vertex->getOutgoingEdges().size() - 1);

    /*for(auto rootVertex: species){
        if(dynamic_cast<const TPG::TPGActivationVertex*>(rootVertex) == nullptr){
            throw std::runtime_error("A root should always be an activation vertex");
        }
        auto currentVertex = (const TPG::TPGActivationVertex*)graph.getVerticesOfRoot(rootVertex).at(indexChoosenVertex);

        // Get the edge to remove
        auto edges = currentVertex->getOutgoingEdges();
        auto itEdges = edges.begin();
        std::advance(itEdges, edgeIndex);
        graph.removeEdge(*(*itEdges));
    }*/

    return true;

    
}
void Mutator::TPGMutator::changeActionClassSpecies(TPG::TPGGraph& graph, 
    const TPG::TPGVertex* species,
    std::list<std::shared_ptr<Program::Program>>& newPrograms,
    const Mutator::MutationParameters& params, Mutator::RNG& rng)
{
    
}
void Mutator::TPGMutator::extendSpecies(TPG::TPGGraph& graph, 
    const TPG::TPGVertex* species,
    std::list<std::shared_ptr<Program::Program>>& newPrograms,
    const Mutator::MutationParameters& params, Mutator::RNG& rng)
{

    double probaExtendActionEdge = 0.7;

    const std::list<const TPG::TPGAgent*>& agents = graph.getAgentsOfSpecies(*species);

    if(dynamic_cast<const TPG::TPGActivationVertex*>(species) == nullptr){
        throw std::runtime_error("A root should always be an activation vertex");
    }
    // All individual of a species should have the exact same structure.
    // Get the first individual to know the shape of the agent
    std::vector<const TPG::TPGVertex *> vertices = graph.getVerticesOfRoot((const TPG::TPGActivationVertex*)species);
    std::vector<const TPG::TPGVertex *> weightedVertices;

    for (const auto* vertex : vertices) {

        // If Vertex is activation vertex, add i N times, with N is the number of action in the vertex
        if (auto activationVertex = dynamic_cast<const TPG::TPGActivationVertex*>(vertex)) {
            weightedVertices.insert(weightedVertices.end(), std::pow(activationVertex->getOutgoingActionEdges().size(),2), vertex);
        }
    }

    // Randomly select an activation vertex to do the extension on based on the weight
    const TPG::TPGActivationVertex* vertexChoosen = (const TPG::TPGActivationVertex*)weightedVertices.at(rng.getUnsignedInt64(0, weightedVertices.size() - 1));

    // Create a decision vertex.
    const TPG::TPGDecisionVertex& decVertex = graph.addNewDecisionVertex();

    // Create two activation vertices
    const TPG::TPGActivationVertex& actVertex1 = graph.addNewActivationVertex();
    const TPG::TPGActivationVertex& actVertex2 = graph.addNewActivationVertex();

    // Add a connexion edge between the current vertex and the decision vertex
    graph.addNewConnectionEdge(*vertexChoosen, decVertex);

    // Connect each context program to one team
    const TPG::TPGDecisionEdge& decisionEdge1 = graph.addNewDecisionEdge(decVertex, actVertex1);
    const TPG::TPGDecisionEdge& decisionEdge2 = graph.addNewDecisionEdge(decVertex, actVertex2);

    // Add new programs to all the agents with the two new edges.
    for(auto agent: agents){
        // Create two random programs
        std::shared_ptr<Program::Program> program1 = std::make_shared<Program::Program>(graph.getEnvironment(), false);
        std::shared_ptr<Program::Program> program2 = std::make_shared<Program::Program>(graph.getEnvironment(), false);
        Mutator::ProgramMutator::initRandomProgram(*program1, params, rng);
        Mutator::ProgramMutator::initRandomProgram(*program2, params, rng);

        // Add the random programs to the agent
        graph.setProgramToAgent(*agent, &decisionEdge1, program1);
        graph.setProgramToAgent(*agent, &decisionEdge2, program2);
    }

    // Randomly choose some action edges and add it to the new activation vertices.
    double proba = 1;
    std::list<TPG::TPGEdge *> actionEdges = vertexChoosen->getOutgoingActionEdges();
    while(actionEdges.size() > 0 && proba > rng.getDouble(0, 1)){
        
        // Get a random edge.
        auto itEdges = actionEdges.begin();
        std::advance(itEdges, rng.getUnsignedInt64(0, actionEdges.size()-1));
        TPG::TPGActionEdge* edge = (TPG::TPGActionEdge*)(*itEdges);


        // Create to new action edges.
        const TPG::TPGActionEdge& actionEdge1 = graph.addNewActionEdge(actVertex1, edge->getActionClass());
        const TPG::TPGActionEdge& actionEdge2 = graph.addNewActionEdge(actVertex2, edge->getActionClass());

        for(auto agent: agents){

            // Get the program of the edge, then duplicate it to create a new program
            std::shared_ptr<Program::Program> actionProg1 = agent->getProgramSharedPointer(edge);
            std::shared_ptr<Program::Program> actionProg2(new Program::Program(*actionProg1, true));
            newPrograms.push_back(actionProg2);

            // Add the two programs to the newly created action edges
            graph.setProgramToAgent(*agent, &actionEdge1, actionProg1);
            graph.setProgramToAgent(*agent, &actionEdge2, actionProg2);
            
            // Remove the program from the former edge.
            graph.removeProgramToAgent(*agent, edge);

        }

        // Remove the older edge.
        graph.removeEdge(*edge);


        // Erase the edge and increase probability
        actionEdges.erase(itEdges);
        proba *= probaExtendActionEdge;


    }
}

const TPG::TPGVertex* Mutator::TPGMutator::moveAgentSpecies(TPG::TPGGraph& graph, 
    const TPG::TPGVertex* species,
    const Mutator::MutationParameters& params, Mutator::RNG& rng)
{
    // Create the new species
    const TPG::TPGVertex& newSpecies = graph.cloneVertex(*species);
    graph.addSpecies(newSpecies);

    // Since the new species has just been copied from the old species, the order of the edges is the same.
    std::vector<TPG::TPGEdge *> oldEdges = graph.getEdgesOfRoot(species, false);
    std::vector<TPG::TPGEdge *> newEdges = graph.getEdgesOfRoot(&newSpecies, false);

    std::list<const TPG::TPGAgent*> initialAgents = graph.getAgentsOfSpecies(*species);
    std::list<const TPG::TPGAgent*> movedAgents;

    size_t wanted_size = initialAgents.size() / 4;

    // Select randomly agents from the initial list.
    while(movedAgents.size() < wanted_size){

        auto it = initialAgents.begin();
        std::advance(it, rng.getUnsignedInt64(0, initialAgents.size() - 1));

        movedAgents.push_back(*it);
        initialAgents.erase(it);
    }
    

    for(const TPG::TPGAgent* agent: movedAgents){



        auto itOldEdges = oldEdges.begin();
        for(TPG::TPGEdge* newEdge: newEdges){
            TPG::TPGEdge* oldEdge = *itOldEdges;
            

            // Add the new edge to the agent and remove the older one.
            graph.setProgramToAgent(*agent, newEdge, agent->getProgramSharedPointer(oldEdge));
            graph.removeProgramToAgent(*agent, oldEdge);

            itOldEdges++;
        }

        // Change the species of the agent
        graph.changeSpecies(*agent, newSpecies);
    }

    return &newSpecies;

}

void Mutator::TPGMutator::mutateSpecies(TPG::TPGGraph& graph, 
    const TPG::TPGVertex* species,
    std::list<std::shared_ptr<Program::Program>>& newPrograms,
    const Mutator::MutationParameters& params, Mutator::RNG& rng)
{

    

    
    const TPG::TPGVertex* newSpecies = moveAgentSpecies(graph, species, params, rng);
    

    double probaAddEdge = 0.0;
    double probaDeletionEdge = 0.0;
    double probaChangeActionClass = 0.0;
    double probaExtension = 1.0;

    bool success = false;
    while(!success){
        double mutationValue = rng.getDouble(0.0, probaAddEdge + probaDeletionEdge + probaChangeActionClass + probaExtension);


        if(probaAddEdge > mutationValue){
            std::cout<<" Add  ";
            success = addEdgeSpecies(graph, newSpecies, newPrograms, params, rng);

            if(!success){
                probaAddEdge = 0.0;
            }
        } else if(probaAddEdge + probaDeletionEdge > mutationValue){
            std::cout<<" Delete  ";
            success = deleteEdgeSpecies(graph, newSpecies, newPrograms, params, rng);
            if(!success){
                probaDeletionEdge = 0.0;
            }
        } else if(probaAddEdge + probaDeletionEdge + probaChangeActionClass > mutationValue){
            changeActionClassSpecies(graph, newSpecies, newPrograms, params, rng);
        } else {
            std::cout<<" Extend  ";
            extendSpecies(graph, newSpecies, newPrograms, params, rng);
            success = true;
        }

    }

    graph.updateAssessedActions(newSpecies);


}
 

void Mutator::TPGMutator::mutateTPGVertex(
    TPG::TPGGraph& graph, const TPG::TPGAgent& agent,
    std::list<std::shared_ptr<Program::Program>>& newPrograms,
    const Mutator::MutationParameters& params, Mutator::RNG& rng)
{

    auto agentPrograms = agent.getPrograms();


    bool anyMutationDone = false;
    do {
        std::vector<uint64_t> indexUsed;
        uint64_t index;
        // 4. mutate randomly selected program on action Edge. 
        double proba = params.tpg.pMutateActionProgram;
        while(indexUsed.size() < agentPrograms.size()  && proba > rng.getDouble(0.0, 1.0)){

            // Search an index not alreay used
            do {
                index = rng.getUnsignedInt64(0, agentPrograms.size()-1);
            } while(std::find(indexUsed.begin(), indexUsed.end(), index) != indexUsed.end()) ;

            // Save the index to avoid using it again
            indexUsed.push_back(index);
    
            // Get the search pair
            auto iter = agentPrograms.begin();
            std::advance(iter, index);
            auto pair = *iter;

            // copy program
            std::shared_ptr<Program::Program> newProg(
                new Program::Program(*pair.second, pair.second->isActionProgram()));

            // Add it to the list of new Program to be mutated.
            newPrograms.push_back(newProg);

            // Set the new program to this agent
            graph.setProgramToAgent(agent, pair.first, pair.second);

            // Decrease the probability of mutation a new edge.
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
    std::vector<const TPG::TPGAgent*>& childs,
    TPG::TPGEdge* edge,
    const Mutator::MutationParameters& params,
    Mutator::RNG& rng)
{

    bool actionProgram = childs.at(0)->getProgramSharedPointer(edge)->isActionProgram();

    // Create new empty programs
    std::array<std::shared_ptr<Program::Program>, 2> newProgs = {
        std::make_shared<Program::Program>(graph.getEnvironment(), actionProgram),
        std::make_shared<Program::Program>(graph.getEnvironment(), actionProgram)
    };

    // Get the programs of the parents, it should alreay be checked that program exist.
    std::array<std::shared_ptr<Program::Program>, 2> originProgs = {
        childs.at(0)->getProgramSharedPointer(edge),
        childs.at(1)->getProgramSharedPointer(edge)
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
        graph.setProgramToAgent(*childs.at(i), edge, newProgs[i]);
        newProgs[i]->identifyIntrons();
    }

}

void Mutator::TPGMutator::crossEdges(
    TPG::TPGGraph& graph,
    std::vector<const TPG::TPGAgent*>& childs,
    std::vector<TPG::TPGEdge*>& edges,
    const Mutator::MutationParameters& params,
    Mutator::RNG& rng)
{
    // For all the edges crossed, exchange the programs.
    for(auto edge: edges){
        std::shared_ptr<Program::Program> prog0 = childs.at(0)->getProgramSharedPointer(edge);

        graph.setProgramToAgent(*childs.at(0), edge, childs.at(1)->getProgramSharedPointer(edge));
        graph.setProgramToAgent(*childs.at(1), edge, prog0);
    }


}

void Mutator::TPGMutator::crossTPGAgents(
    TPG::TPGGraph& graph,
    std::vector<const TPG::TPGAgent*>& childs,
    const Mutator::MutationParameters& params,
    Mutator::RNG& rng)
{




    if(params.tpg.probaCrossAgents < rng.getDouble(0.0, 1.0)){
        return;
    }
    

    // Get all the edges of the species.
    std::vector<TPG::TPGEdge*> edges = graph.getEdgesOfRoot(childs.at(0)->getRootSpecies());

    // Select a random edge.
    TPG::TPGEdge* selectedEdge = edges.at(rng.getUnsignedInt64(0, edges.size() - 1));

    // Get all the crossed edges.
    std::vector<TPG::TPGEdge*> edgeCrossed;
    if(dynamic_cast<TPG::TPGActionEdge*>(selectedEdge) == nullptr){
        edgeCrossed = graph.getEdgesOfRoot(selectedEdge->getDestination(), false);
    }
    if(dynamic_cast<TPG::TPGConnectionEdge*>(selectedEdge) == nullptr){
        edgeCrossed.push_back(selectedEdge);
    }

    crossEdges(graph, childs, edgeCrossed, params, rng);



    std::vector<uint64_t> indicesUsed;
    uint64_t currentIndex;

    // Always do at least one crossover, except is the proba is at zero (mearning we don't want any crossover)
    double proba = params.tpg.probaCrossPrograms;
    while(indicesUsed.size() < edgeCrossed.size()  && proba > rng.getDouble(0.0, 1.0)){


        // Select the edge index
        do {
            currentIndex = rng.getUnsignedInt64(0, edgeCrossed.size()-1);
        } while(std::find(indicesUsed.begin(), indicesUsed.end(), currentIndex) != indicesUsed.end()) ;

        indicesUsed.push_back(currentIndex);

        if(params.tpg.typeProgramCrossover == "standard"){
            crossProgram(graph, childs, edgeCrossed.at(currentIndex), params, rng);
        } else {
            throw std::runtime_error("params.mutation.tpg.typeProgramCrossover not found");
        }



        proba *= params.tpg.probaCrossPrograms;
    }




    


}

void Mutator::TPGMutator::populateTPG(TPG::TPGGraph& graph,
                                      const Archive& archive,
                                      const Mutator::MutationParameters& params,
                                      Mutator::RNG& rng, uint64_t nbActions,
                                      uint64_t maxNbThreads)
{
    // Create an empty list to store Programs to mutate.
    std::list<std::shared_ptr<Program::Program>> newPrograms;

    // Get the number of agents.
    size_t currentNumberOfAgents = graph.getNbAgents();


    for(const TPG::TPGVertex* species: graph.getRootVertices()){
        std::cout<<"size species "<<graph.getAgentsOfSpecies(*species).size()<<std::endl;
    }

    for(const TPG::TPGVertex* species: graph.getRootVertices()){


        // Get current vertex set (copy)
        std::list<const TPG::TPGAgent *> agents(graph.getAgentsOfSpecies(*species));

        if(agents.size()>10){
            // Get the current number of agents
            size_t currentSizeOfSpecies = agents.size();

            bool useTournamentSelection = graph.getEnvironment().getParams().useTournamentSelection;
            if (useTournamentSelection) {
                // The root not set to be deleted are not used during evolution
                agents.erase(
                    std::remove_if(agents.begin(), agents.end(),
                                [](const TPG::TPGAgent* agent) -> bool {
                                    return !agent->isToBeDeleted();}),
                                    agents.end());
            }


            // Get the expected size of the species and compute the number of agents to create
            uint64_t expectedSizeOfSpecies = params.tpg.nbRoots * currentSizeOfSpecies / currentNumberOfAgents;
            uint64_t nbAgentsToCreate = expectedSizeOfSpecies - currentSizeOfSpecies + (agents.size() * useTournamentSelection);


            std::vector<const TPG::TPGAgent*> agentsParents1(agents.begin(), agents.end());
            std::vector<const TPG::TPGAgent*> agentsParents2;
            if(useTournamentSelection){
                // Divide root used into two subVector with half of the roots, randomly selected.
                for(size_t idx = 0; idx < agents.size() / 2; idx++){
                    auto root = agentsParents1.at(rng.getUnsignedInt64(0, agentsParents1.size() - 1));
            
                    agentsParents2.push_back(root);
                    std::swap(root, agentsParents1.back());
                    agentsParents1.pop_back();
                }
            } else {
                agentsParents2 = std::vector<const TPG::TPGAgent*>(agents.begin(), agents.end());
            }

            uint64_t nbAgentsCreated = 0;
            while (nbAgentsCreated < nbAgentsToCreate) {

                // Not really clean but efficient switching between tournament and not tournament selection
                // Select a random existing root
                uint64_t clonedRootIndex1 =
                    rng.getUnsignedInt64(0, agentsParents1.size() - 1);
                // Select a random existing root
                uint64_t clonedRootIndex2 =
                    rng.getUnsignedInt64(0, agentsParents2.size() - 2 + useTournamentSelection);
                
                // Be sure it is different if we do not use tournament selection
                if(clonedRootIndex1 == clonedRootIndex2 && !useTournamentSelection){
                    clonedRootIndex2++;
                }

                const TPG::TPGAgent* child1 = &graph.cloneAgent(*agentsParents1.at(clonedRootIndex1));
                const TPG::TPGAgent* child2 = &graph.cloneAgent(*agentsParents2.at(clonedRootIndex2));

                // Get parents and create childs
                std::vector<const TPG::TPGAgent*> childs{child1, child2};

                // Do the crossover over the childs
                crossTPGAgents(graph, childs, params, rng);

                // Do the mutation over the childs
                for(auto child: childs){
                    mutateTPGVertex(graph, *child, newPrograms, params, rng);
                }
                // Check the new number of roots
                // Needed since preExisting root may be subsumed by new ones.
                nbAgentsCreated += 2;
            }



            for(auto agent: agents){
                if(agent->isToBeDeleted()){
                    graph.removeAgent(*agent);
                }
            }
        } else {
            for(auto agent: agents){
                graph.removeAgent(*agent);
            }
            graph.removeSpecies(*species);
            graph.removeVertex(*species);
        }



    }


    double probaMutateSpecies = 1.0;
    if(probaMutateSpecies > rng.getDouble(0, 1) && nbActions > 0 && nbActions < 10){

        std::vector<const TPG::TPGVertex*> species(graph.getRootVertices());

        std::cout<<"\n"<<species.size()<<std::endl;
        // The root not set to be deleted are not used during evolution
        species.erase(
            std::remove_if(species.begin(), species.end(),
                        [&graph](const TPG::TPGVertex* root) -> bool {
                            return graph.getNbAgentsOfSpecies(*root) < 100;}),
                            species.end());

        std::cout<<"\n"<<species.size()<<std::endl;

        if(species.size() > 0){
            size_t index = rng.getUnsignedInt64(0, species.size()- 1);
            std::cout<<"Index hchoosen "<<index<<std::endl;
            // Dupplicate species and copy the new agents
            mutateSpecies(graph, species.at(index), newPrograms, params, rng);
        }
    }

    // Mutate the new Programs
    mutateNewProgramBehaviors(maxNbThreads, newPrograms, rng, params, archive);
}
