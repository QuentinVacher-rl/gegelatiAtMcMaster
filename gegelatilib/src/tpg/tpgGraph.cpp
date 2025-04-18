/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019 - 2024) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019 - 2023)
 * Nicolas Sourbier <nsourbie@insa-rennes.fr> (2019 - 2020)
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

#include <queue>

#include <algorithm>
#include <stdexcept>
#include <type_traits>

#include "tpg/tpgGraph.h"

TPG::TPGGraph::~TPGGraph()
{
    // Just delete the vertices.
    // Edges will be deleted during their container destruction.
    for (TPG::TPGVertex* vertex : this->vertices) {
        delete vertex;
    }
}

TPG::TPGGraph& TPG::TPGGraph::operator=(TPGGraph model)
{
    swap(*this, model);
    return *this;
}

void TPG::TPGGraph::clear()
{
    // Remove all vertices
    while (this->vertices.size() > 0) {
        this->removeVertex(*this->vertices.front());
    }
}

const Environment& TPG::TPGGraph::getEnvironment() const
{
    return this->env;
}

const TPG::TPGFactory& TPG::TPGGraph::getFactory() const
{
    return *this->factory;
}

const TPG::TPGDecisionVertex& TPG::TPGGraph::addNewDecisionVertex()
{
    this->vertices.push_back(factory->createTPGDecisionVertex());
    return (const TPGDecisionVertex&)(*this->vertices.back());
}

const TPG::TPGActivationVertex& TPG::TPGGraph::addNewActivationVertex()
{
    this->vertices.push_back(factory->createTPGActivationVertex());
    return (const TPGActivationVertex&)(*this->vertices.back());
}

const TPG::TPGAgent& TPG::TPGGraph::addNewAgent(const TPG::TPGVertex& root)
{
    
    auto vertexIterator = this->findVertex(&root);
    if (vertexIterator == this->vertices.end()) {
        throw std::runtime_error(
            "The root search does not exist in the TPGGraph.");
    }

    // Get the root of the species
    TPG::TPGVertex* rootSpecies = *vertexIterator;


    // Check that the root is not already in the species map
    if(this->species.find(rootSpecies) != this->species.end()){
        TPG::TPGAgent* agent = factory->createTPGAgent(&root);
        this->species.at(rootSpecies).push_back(agent);
        this->agents.push_back(agent);

        return *agent;
    } else {
        throw std::runtime_error(
            "Can not add a TPGAgent to a species that does not exist.");
    }
}


void TPG::TPGGraph::addSpecies(const TPG::TPGVertex& root)
{
        
    auto vertexIterator = this->findVertex(&root);
    if (vertexIterator == this->vertices.end()) {
        throw std::runtime_error(
            "The root search does not exist in the TPGGraph.");
    }

    // Get the root of the species
    TPG::TPGVertex* rootSpecies = *vertexIterator;

    // Check that the root is not already in the species map
    if(this->species.find(rootSpecies) == this->species.end()){
        this->species.insert(std::make_pair(rootSpecies, std::list<const TPG::TPGAgent*>()));
    } else {
        throw std::runtime_error(
            "The root species already exist in the map of species");
    }
}

void TPG::TPGGraph::removeSpecies(const TPG::TPGVertex& root)
{
        
    auto vertexIterator = this->findVertex(&root);
    if (vertexIterator == this->vertices.end()) {
        throw std::runtime_error(
            "The root search does not exist in the TPGGraph.");
    }

    // Get the root of the species
    TPG::TPGVertex* rootSpecies = *vertexIterator;

    // Check that the root exist in the map of species
    if(this->species.find(rootSpecies) != this->species.end()){
        
        // Delete the TPGAgents;
        for(auto agent: this->species.at(rootSpecies)){
            this->removeAgent(*agent);
        }
        this->species.erase(rootSpecies);
    } else {
        throw std::runtime_error(
            "Cannot delete a root species that does not exist.");
    }
}

size_t TPG::TPGGraph::getNbVertices() const
{
    return this->vertices.size();
}

const std::vector<const TPG::TPGVertex*> TPG::TPGGraph::getVertices() const
{
    std::vector<const TPG::TPGVertex*> result(this->vertices.size());
    std::copy(this->vertices.begin(), this->vertices.end(), result.begin());
    return result;
}

uint64_t TPG::TPGGraph::getNbRootVertices() const
{
    return std::count_if(this->vertices.begin(), this->vertices.end(),
                         [](const TPGVertex* vertex) {
                             return vertex->getIncomingEdges().size() == 0;
                         });
}

const std::vector<const TPG::TPGVertex*> TPG::TPGGraph::getRootVertices() const
{
    std::vector<const TPG::TPGVertex*> result;
    std::copy_if(this->vertices.begin(), this->vertices.end(),
                 std::back_inserter(result), [](TPGVertex* vertex) {
                     return vertex->getIncomingEdges().size() == 0;
                 });
    return result;
}

uint64_t TPG::TPGGraph::getNbAgents() const
{
    return this->agents.size();
}

const std::vector<const TPG::TPGAgent*> TPG::TPGGraph::getAgents() const {

    std::vector<const TPG::TPGAgent*> result(this->agents.size());
    std::copy(this->agents.begin(), this->agents.end(), result.begin());
    return result;
}

const std::list<const TPG::TPGAgent*>& TPG::TPGGraph::getAgentsOfSpecies(const TPG::TPGVertex& root)
{
    
    auto vertexIterator = this->findVertex(&root);
    if (vertexIterator == this->vertices.end()) {
        throw std::runtime_error(
            "The root search for species does not exist in the TPGGraph.");
    }

    if(this->species.find(&root) == this->species.end()){
        throw std::runtime_error("Cannot find the agent of a species not in the graph");
    }
    return this->species.at(&root);
}


uint64_t TPG::TPGGraph::getNbAgentsOfSpecies(const TPG::TPGVertex& root)
{
    auto vertexIterator = this->findVertex(&root);
    if (vertexIterator == this->vertices.end()) {
        throw std::runtime_error(
            "The root search for species does not exist in the TPGGraph.");
    }

    if(this->species.find(&root) == this->species.end()){
        throw std::runtime_error("Cannot find the agent of a species not in the graph");
    }
    return this->species.at(&root).size();
}

bool TPG::TPGGraph::hasVertex(const TPG::TPGVertex& vertex) const
{
    return std::find(this->vertices.cbegin(), this->vertices.cend(), &vertex) !=
           this->vertices.cend();
}

void TPG::TPGGraph::removeVertex(const TPGVertex& vertex)
{
    // Remove the vertex based on a pointer comparison.
    auto iterator = this->findVertex(&vertex);
    if (iterator != this->vertices.end()) {
        // Remove all connected edges.
        // copy inEdges set for removal
        // (because iterating on the modified set is not a good idea).
        std::list<TPGEdge*> inEdgesToRemove = (*iterator)->getIncomingEdges();
        for (auto inEdge : inEdgesToRemove) {
            this->removeEdge(*inEdge);
        }
        // copy outEdges set for removal
        std::list<TPGEdge*> outEdgesToRemove = (*iterator)->getOutgoingEdges();
        for (auto outEdge : outEdgesToRemove) {
            this->removeEdge(*outEdge);
        }

    }

    // Remove edge for action can launch again remove vertex.
    // Check again if the vertex is in the graph before deleting.
    iterator = this->findVertex(&vertex);
    if (iterator != this->vertices.end()) {
        // Free the memory of the vertex
        delete *iterator;
        // Remove the pointer from the list.
        this->vertices.erase(iterator);
    }
}


const TPG::TPGVertex& TPG::TPGGraph::cloneVertex(const TPGVertex& vertex)
{
    // Check that the vertex to clone exists in the graph
    auto vertexIterator = this->findVertex(&vertex);
    if (vertexIterator == this->vertices.end()) {
        throw std::runtime_error(
            "The vertex to clone does not exist in the TPGGraph.");
    }

    // Create a new Vertex
    // (at the end of the vertices list)
    if (dynamic_cast<const TPG::TPGDecisionVertex*>(&vertex) != nullptr) {
        this->addNewDecisionVertex();
    }
    else if (dynamic_cast<const TPG::TPGActivationVertex*>(&vertex) != nullptr) {
        this->addNewActivationVertex();
    }

    // Get the new vertex
    TPGVertex* newVertex = this->vertices.back();

    // Copy the outgoing edges, and their destination (if any).
    for (auto edge : vertex.getOutgoingEdges()) {

        if(dynamic_cast<TPG::TPGDecisionEdge*>(edge) != nullptr){
            const TPGVertex& destinationVertex = this->cloneVertex(*edge->getDestination());
            this->addNewDecisionEdge(*newVertex, destinationVertex);
        } else if (dynamic_cast<TPG::TPGConnectionEdge*>(edge) != nullptr) {
            const TPGVertex& destinationVertex = this->cloneVertex(*edge->getDestination());
            this->addNewConnectionEdge(*newVertex, destinationVertex);
        } else {
            TPG::TPGActionEdge* actionEdge = dynamic_cast<TPGActionEdge*>(edge);
            this->addNewActionEdge(*newVertex,
                                   actionEdge->getActionClass());
        }

    }

    newVertex->updateAssessedActions();

    return *newVertex;
}

bool TPG::TPGGraph::hasAgent(const TPG::TPGAgent& agent)
{
    auto iterator = this->findAgent(&agent);
    return iterator != this->agents.end();
}

void TPG::TPGGraph::removeAgent(const TPGAgent& agent)
{
    // Find the agent
    auto iterator = this->findAgent(&agent);
    if(iterator != this->agents.end()){

        // Get the root of the agent
        auto root = agent.getRootSpecies();
        if(this->species.find(root) == this->species.end()){
            throw std::runtime_error("Cannot find the agent of a species not in the graph");
        }
        // Get the corresponding list of agent of the species
        std::list<const TPG::TPGAgent *>& listAgents = this->species.at(root);

        // Search the agent in the list and erase it
        auto iteratorSpecies = std::find(listAgents.begin(), listAgents.end(), &agent);
        if(iteratorSpecies != listAgents.end()){
            listAgents.erase(iteratorSpecies);
        }
        
        // Delete the agent and erase it from the list of all agents
        delete *iterator;
        this->agents.erase(iterator);
    }

}

const TPG::TPGAgent& TPG::TPGGraph::cloneAgent(const TPGAgent& agent)
{

    auto root = agent.getRootSpecies();
    if(this->species.find(root) == this->species.end()){
        throw std::runtime_error("Cannot find the agent of a species not in the graph");
    }

    auto iterator = this->findAgent(&agent);
    if(iterator == this->agents.end()){
        throw std::runtime_error("Cannot clone an agent not in the graph");
    }

    // Create a new agent and add each program
    const TPGAgent& newAgent = this->addNewAgent(*root);
    for(auto pair: (*iterator)->getPrograms()){
        this->setProgramToAgent(newAgent, pair.first, pair.second);
    }

    return newAgent;

}

void TPG::TPGGraph::changeSpecies(const TPG::TPGAgent& agent, const TPG::TPGVertex& root)
{
    // For new species
    auto rootIterator = this->findVertex(&root);
    if (rootIterator == this->vertices.end()) {
        throw std::runtime_error(
            "The root search for species does not exist in the TPGGraph.");
    }

    if(this->species.find(&root) == this->species.end()){
        throw std::runtime_error("Cannot find the agent of a species not in the graph");
    }

    auto iterator = this->findAgent(&agent);
    if(iterator == this->agents.end()){
        throw std::runtime_error("Cannot clone an agent not in the graph");
    }

    // For origin species
    const TPG::TPGVertex* origSpecies = (*iterator)->getRootSpecies();
    auto origRootIterator = this->findVertex(origSpecies);
    if (origRootIterator == this->vertices.end()) {
        throw std::runtime_error(
            "The root search for species does not exist in the TPGGraph.");
    }

    if(this->species.find(origSpecies) == this->species.end()){
        throw std::runtime_error("Cannot find the agent of a species not in the graph");
    }

    // Remove the agent from the original list of agents.
    std::list<const TPG::TPGAgent *>& listOrigAgents = this->species.at(origSpecies);
    auto origIt = std::find(listOrigAgents.begin(), listOrigAgents.end(), &agent);
    if(origIt != listOrigAgents.end()){
        listOrigAgents.erase(origIt);
    } else {
        throw std::runtime_error("Cannot remove an agent from a species that it don't belong to");
    }

    // Set the new root species
    (*iterator)->setRootSpecies(&root);

    // Add the agent to the new species.
    this->species.at(&root).push_back(&agent);
    
}


void TPG::TPGGraph::setProgramToAgent(const TPGAgent& agent, const TPG::TPGEdge* edge, std::shared_ptr<Program::Program> prog)
{
    auto iterator = this->findAgent(&agent);
    if(iterator == this->agents.end()){
        throw std::runtime_error("Cannot set a program to an agent not in the graph");
    }

    (*iterator)->setProgram(edge, prog);
}

void TPG::TPGGraph::removeProgramToAgent(const TPGAgent& agent, const TPG::TPGEdge* edge)
{
    auto iterator = this->findAgent(&agent);
    if(iterator == this->agents.end()){
        throw std::runtime_error("Cannot remove a program from an agent not in the graph");
    }

    (*iterator)->deletePair(edge);
}

const TPG::TPGDecisionEdge& TPG::TPGGraph::addNewDecisionEdge(
    const TPGVertex& src, const TPGVertex& dest)
{
    // Check the TPGVertex existence within the graph.
    auto srcVertex =
        std::find_if(this->vertices.begin(), this->vertices.end(),
                     [&src](TPG::TPGVertex* other) { return other == &src; });
    auto dstVertex =
        std::find_if(this->vertices.begin(), this->vertices.end(),
                     [&dest](TPG::TPGVertex* other) { return other == &dest; });
    if (dstVertex == this->vertices.end() ||
        srcVertex == this->vertices.end()) {
        throw std::runtime_error("Attempting to add a TPGEdge between vertices "
                                 "not present in the TPGGraph.");
    }

    // Create the edge
    this->edges.push_back(factory->createTPGDecisionEdge(&src, &dest));
    TPGDecisionEdge& newEdge = *((TPGDecisionEdge*)this->edges.back().get());

    // Add the edged to the Vertices
    try {
        // (May throw if an outgoing edge is added to an action)
        (*srcVertex)->addOutgoingEdge(&newEdge);
    }
    catch (std::runtime_error& e) {
        // Remove the edge before re-throwing
        this->edges.pop_back();
        throw e;
    }
    (*dstVertex)->addIncomingEdge(&newEdge);

    // return the new edge
    return newEdge;
}

const TPG::TPGActionEdge& TPG::TPGGraph::addNewActionEdge(
    const TPGVertex& src, uint64_t actionClass)
{
    // Check the TPGVertex existence within the graph.
    auto srcVertex =
        std::find_if(this->vertices.begin(), this->vertices.end(),
                     [&src](TPG::TPGVertex* other) { return other == &src; });
    if (srcVertex == this->vertices.end()) {
        throw std::runtime_error(
            "Attempting to add a TPGActionEdge with a vertex "
            "not present in the TPGGraph.");
    }

    // Create the edge
    this->edges.push_back(
        factory->createTPGActionEdge(&src, actionClass));
    TPGActionEdge& newEdge = *((TPGActionEdge*)this->edges.back().get());

    (*srcVertex)->addOutgoingEdge(&newEdge);

    // return the new edge
    return newEdge;
}

const TPG::TPGConnectionEdge& TPG::TPGGraph::addNewConnectionEdge(
    const TPGVertex& src, const TPGVertex& dest)
{
    // Check the TPGVertex existence within the graph.
    auto srcVertex =
        std::find_if(this->vertices.begin(), this->vertices.end(),
                     [&src](TPG::TPGVertex* other) { return other == &src; });
    auto dstVertex =
        std::find_if(this->vertices.begin(), this->vertices.end(),
                     [&dest](TPG::TPGVertex* other) { return other == &dest; });
    if (dstVertex == this->vertices.end() ||
        srcVertex == this->vertices.end()) {
        throw std::runtime_error("Attempting to add a TPGEdge between vertices "
                                 "not present in the TPGGraph.");
    }

    // Create the edge
    this->edges.push_back(factory->createTPGConnectionEdge(&src, &dest  ));
    TPGConnectionEdge& newEdge = *((TPGConnectionEdge*)this->edges.back().get());

    // Add the edged to the Vertices
    try {
        // (May throw if an outgoing edge is added to an action)
        (*srcVertex)->addOutgoingEdge(&newEdge);
    }
    catch (std::runtime_error& e) {
        // Remove the edge before re-throwing
        this->edges.pop_back();
        throw e;
    }
    (*dstVertex)->addIncomingEdge(&newEdge);

    // return the new edge
    return newEdge;
}

const std::list<std::unique_ptr<TPG::TPGEdge>>& TPG::TPGGraph::getEdges() const
{
    return this->edges;
}

void TPG::TPGGraph::removeEdge(const TPGEdge& edge)
{

    // Get the edge (if it is in the graph)
    auto iterator = std::find_if(this->edges.begin(), this->edges.end(),
                                 [&edge](std::unique_ptr<TPG::TPGEdge>& other) {
                                     return &edge == other.get();
                                 });

    // Disconnect the edge from the vertices
    if (iterator == this->edges.end()) {
        throw std::runtime_error(
            "Cannot erase a edge that does not belong to the graph");
    }


    // Remove the edge from the source
    (*this->findVertex(iterator->get()->getSource()))
        ->removeOutgoingEdge(iterator->get());

    // Remove the edge from the destination if it is not an action edge
    if (dynamic_cast<const TPGActionEdge*>(iterator->get()) == nullptr) {

        auto destination = iterator->get()->getDestination();
        (*this->findVertex(destination))->removeIncomingEdge(iterator->get());

        // If destination has 0 incomming edge, remove destination
        if(destination->getIncomingEdges().size() == 0){
            this->removeVertex(*destination);
        }
    }


    // Remove the edge
    this->edges.erase(iterator);
}


const TPG::TPGEdge& TPG::TPGGraph::cloneEdge(const TPGEdge& edge)
{
    auto iterEdge = findEdge(&edge);
    if (iterEdge == this->edges.end()) {
        throw std::runtime_error(
            "Cannot duplicate an Edge not belonging to the graph.");
    }
    else if (dynamic_cast<const TPGActionEdge*>(iterEdge->get()) != nullptr) {
        const TPG::TPGActionEdge* actionEdge =
            dynamic_cast<const TPGActionEdge*>(iterEdge->get());
        return this->addNewActionEdge(*actionEdge->getSource(),
                                      actionEdge->getActionClass());
    }
    else if (dynamic_cast<const TPGConnectionEdge*>(iterEdge->get()) != nullptr){
        return this->addNewConnectionEdge(*iterEdge->get()->getSource(),
                                *iterEdge->get()->getDestination());
    } else {
        return this->addNewDecisionEdge(*iterEdge->get()->getSource(),
                                        *iterEdge->get()->getDestination());

    }
}

bool TPG::TPGGraph::setEdgeDestination(const TPGEdge& edge,
                                       const TPGVertex& newDest)
{
    // Find the edge and vertex
    auto iterNewDestination = findVertex(&newDest);
    auto iterEdge = findEdge(&edge);
    if (iterNewDestination != this->vertices.end() &&
        iterEdge != this->edges.end()) {
        // Unregister the edge from the old destination
        const TPG::TPGVertex* oldDestination =
            iterEdge->get()->getDestination();
        auto iterOldDest = findVertex(oldDestination);
        // finding the vertex should not fail. Otherwise, the exception for
        // next line would be well deserved since it means an edge in the
        // graph is connected to a vertex not in the graph.
        (*iterOldDest)->removeIncomingEdge(iterEdge->get());
        // Register the edge to the new destination
        (*iterNewDestination)->addIncomingEdge(iterEdge->get());
        // Set the destination
        iterEdge->get()->setDestination(*iterNewDestination);
        return true;
    }
    else {
        return false;
    }
}

bool TPG::TPGGraph::setEdgeSource(const TPGEdge& edge, const TPGVertex& newSrc)
{
    // Find the edge and vertex
    auto iterNewSrc = findVertex(&newSrc);
    auto iterEdge = findEdge(&edge);
    if (iterNewSrc != this->vertices.end() && iterEdge != this->edges.end()) {
        // Unregister the edge from the old source
        const TPG::TPGVertex* oldSrc = iterEdge->get()->getSource();
        auto iterOldSrc = findVertex(oldSrc);
        // finding the vertex should not fail. Otherwise, the exception for
        // next line would be well deserved since it means an edge in the
        // graph is connected to a vertex not in the graph.
        (*iterOldSrc)->removeOutgoingEdge(iterEdge->get());
        // Register the edge to the new source
        (*iterNewSrc)->addOutgoingEdge(iterEdge->get());
        // Set the destination
        iterEdge->get()->setSource(*(iterNewSrc));
        return true;
    }
    else {
        return false;
    }
}

std::list<TPG::TPGVertex*>::iterator TPG::TPGGraph::findVertex(
    const TPG::TPGVertex* vertex)
{
    return std::find(this->vertices.begin(), this->vertices.end(), vertex);
}

std::list<std::unique_ptr<TPG::TPGEdge>>::iterator TPG::TPGGraph::findEdge(
    const TPGEdge* edge)
{
    return std::find_if(this->edges.begin(), this->edges.end(),
                        [&edge](std::unique_ptr<TPG::TPGEdge>& other) {
                            return other.get() == edge;
                        });
}

std::list<TPG::TPGAgent*>::iterator TPG::TPGGraph::findAgent(
    const TPG::TPGAgent* agent)
{
    return std::find(this->agents.begin(), this->agents.end(), agent);
}


void TPG::TPGGraph::clearProgramIntrons()
{
    for (TPG::TPGAgent* agent : this->agents) {
        for(auto pairAgent: agent->getPrograms()){
            pairAgent.second->clearIntrons();
        }
    }
}

void TPG::TPGGraph::setActionClassEdge(const TPGEdge* edge, uint64_t newActionClass)
{
    auto it = this->findEdge(edge);

    if (it != this->edges.end()) {
        if(dynamic_cast<TPG::TPGActionEdge*>(it->get()) == nullptr){
            throw std::runtime_error(
                "Trying to set an action class on a context edge");
        }
        // Found the edge, modify it as needed
        dynamic_cast<TPG::TPGActionEdge*>(it->get())->setActionClass(newActionClass);
    } else {
        throw std::runtime_error(
            "Edges not in the graph.");
    }

}


void TPG::TPGGraph::updateAssessedActions(const TPG::TPGVertex* vertex) {
    std::queue<const TPG::TPGVertex*> vertexToUpdate;
    vertexToUpdate.push(vertex);

    while (!vertexToUpdate.empty()) {
        // Get the front vertex in the queue
        auto currentVertex = vertexToUpdate.front();
        vertexToUpdate.pop();

        // Find the vertex to get the non-const reference
        auto it = this->findVertex(currentVertex);
        if (it != this->vertices.end()) {
            // Add the vertices leading to the current vertex to the queue
            for (auto incomingEdge : (*it)->getIncomingEdges()) {
                vertexToUpdate.push(incomingEdge->getSource());
            }

            // Update assessed actions for the current vertex
            (*it)->updateAssessedActions();
        } else {
            throw std::runtime_error(
                "Vertex to assess actions not in the graph.");
        }
    }
}

void TPG::TPGGraph::updateAllAssessedActions() {

    // Launch update method for all actions. 
    // All teams should be linked to actions, even not directly.
    for(auto vertex: this->vertices){
        if(dynamic_cast<TPGActivationVertex*>(vertex) != nullptr){
            this->updateAssessedActions(vertex);
        }
    }
}

void TPG::TPGGraph::setToBeDeleted(const TPG::TPGAgent& agent){

    auto iterator = this->findAgent(&agent);

    if (iterator != this->agents.end()) {
        // Found the vertex, modify it as needed
        (*iterator)->setToBeDeleted(true);
    } else {
        throw std::runtime_error(
            "Vertex not in the graph.");
    }
}

std::vector<const TPG::TPGVertex*> TPG::TPGGraph::getVerticesOfRoot(const TPG::TPGVertex* root){


	std::vector<const TPG::TPGVertex*> verticesSearch = {root};
    std::vector<const TPG::TPGVertex*> vertices = {root};

    // Search while there is still vertices
	while(verticesSearch.size() > 0){

        // Get the first one, then erase it
		auto currVertex = verticesSearch.front();
		verticesSearch.erase(verticesSearch.begin());

        // For each edge in this vertex
		for(auto edge: currVertex->getOutgoingEdges()){
			if(dynamic_cast<TPG::TPGActionEdge*>(edge) == nullptr){
				verticesSearch.push_back(edge->getDestination());
				vertices.push_back(edge->getDestination());
			}
		}
	}

    return vertices;
}

std::vector<TPG::TPGEdge*> TPG::TPGGraph::getEdgesOfRoot(const TPG::TPGVertex* root, bool getConnectionEdge){
    std::vector<TPG::TPGEdge*> edges;

    // Get all the vertices of this root
    std::vector<const TPG::TPGVertex*> vertices = getVerticesOfRoot(root);

    // Get all the edges.
    for(auto vertex: vertices){
        for(auto edge: vertex->getOutgoingEdges()){

            // If connection edge are collected, or if edge is not a connection edge, collect the edge
            if(getConnectionEdge || dynamic_cast<TPG::TPGConnectionEdge*>(edge) == nullptr){
                edges.push_back(edge);
            }
        }
    }
    return edges;
}

