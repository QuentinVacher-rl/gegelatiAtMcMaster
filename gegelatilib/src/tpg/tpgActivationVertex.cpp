/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019 - 2020) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019)
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

#include "tpg/tpgActivationVertex.h"
#include "tpg/tpgActionEdge.h"
#include "tpg/tpgDecisionEdge.h"
#include "tpg/tpgConnectionEdge.h"
#include <stdexcept>


void TPG::TPGActivationVertex::addIncomingEdge(TPGEdge* edge)
{
    if (dynamic_cast<TPGConnectionEdge*>(edge) != nullptr || dynamic_cast<TPGActionEdge*>(edge) != nullptr) {
        throw std::runtime_error(
            "Can only add an incomming ConnectionEdge to a DecisionVertex.");
    }
    else {
        TPGVertex::addIncomingEdge(edge);
    }
}

void TPG::TPGActivationVertex::addOutgoingEdge(TPGEdge* edge)
{
    if (dynamic_cast<TPGDecisionEdge*>(edge) != nullptr) {
        throw std::runtime_error(
            "Cannot add an outgoing decisionEdge to an ActivationVertex.");
    }
    else {
        TPGVertex::addOutgoingEdge(edge);
    }
}

std::list<TPG::TPGEdge*> TPG::TPGActivationVertex::getOutgoingConnectionEdges() const{

    std::list<TPG::TPGEdge*> connectionEdges;
    std::for_each(outgoingEdges.begin(), outgoingEdges.end(),
                  [&connectionEdges](TPG::TPGEdge* edge) {
                      if (dynamic_cast<TPG::TPGConnectionEdge*>(edge) !=
                          nullptr) {
                            connectionEdges.push_back(edge);
                      }
                  });
    return connectionEdges;
}

std::list<TPG::TPGEdge*> TPG::TPGActivationVertex::getOutgoingActionEdges() const{

    std::list<TPG::TPGEdge*> actionEdges;
    std::for_each(outgoingEdges.begin(), outgoingEdges.end(),
                  [&actionEdges](TPG::TPGEdge* edge) {
                      if (dynamic_cast<TPG::TPGActionEdge*>(edge) !=
                          nullptr) {
                            actionEdges.push_back(edge);
                      }
                  });
    return actionEdges;
}


void TPG::TPGActivationVertex::orderOutgoingEdges() {

    outgoingEdges.sort([](TPG::TPGEdge* edge1, TPG::TPGEdge* edge2) {

        if(dynamic_cast<TPG::TPGActionEdge*>(edge1) != nullptr){
            // both edge are action edge
            if(dynamic_cast<TPG::TPGActionEdge*>(edge2) != nullptr){
                return dynamic_cast<TPG::TPGActionEdge*>(edge1)->getActionClass() < dynamic_cast<TPG::TPGActionEdge*>(edge2)->getActionClass();

            // Edge1 is action edge and edge2 is ConnectionEdge
            } else {
                return false;
            } 
            // both edge are connection edge
        } else if (dynamic_cast<TPG::TPGConnectionEdge*>(edge2) != nullptr){
            return edge1->getDestination()->getPath().back() < edge2->getDestination()->getPath().back();
            // Edge1 is connection edge and edge2 is ActionEdge
        } else {
            return true;
        }
    });
}

TPG::TPGActionEdge* TPG::TPGActivationVertex::getEdgeOfAction(uint64_t actionClass) const {

    auto actionEdges = this->getOutgoingConnectionEdges();

    // Search the edge with the searched action class
    auto it = std::find_if(
        actionEdges.begin(), actionEdges.end(),
        [actionClass](TPG::TPGEdge* edge) {
            return static_cast<TPG::TPGActionEdge*>(edge)->getActionClass() == actionClass;
        }
    );

    // If action founded, return the shared pointer, else return nullptr
    if(it != actionEdges.end()){
        return (TPG::TPGActionEdge*)(*it);
    } else {
        return nullptr;
    }

}

void TPG::TPGActivationVertex::updateAssessedActions()
{
    assessedActions.clear();
    auto actionEdges = this->getOutgoingActionEdges();
    for (TPGEdge* actionEdge : actionEdges) {
        // The edge is an action edge, insert its action class
        assessedActions.insert(((TPGActionEdge*)actionEdge)->getActionClass());

    }

    auto connectionEdges = this->getOutgoingConnectionEdges();
    for (TPGEdge* connectionEdge : connectionEdges) {

        // Insert all assessed actions from the destination
        const auto& destinationActions = connectionEdge->getDestination()->getAssessedActions();
        assessedActions.insert(destinationActions.begin(), destinationActions.end());
    }
}