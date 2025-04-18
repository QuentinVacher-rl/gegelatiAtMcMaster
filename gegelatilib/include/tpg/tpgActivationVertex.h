/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019)
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

#ifndef TPG_ACTIVATION_VERTEX_H
#define TPG_ACTIVATION_VERTEX_H

#include "tpg/tpgVertex.h"

namespace TPG {
    /**
     * Class used to represent an activation vertex of the TPGGraph.
     *
     * An activation vertex is a vertex with only connection or action edges.
     */
    class TPGActivationVertex : public TPGVertex
    {
        public: 

            /**
             * \brief Specialisation of TPGVertex method to accept only DecisionEdges
             *
             * \param[in] edge the TPGEdge pointer to be added to the outgoingEdges
             *                 Set.
             */
            virtual void addIncomingEdge(TPG::TPGEdge* edge);
            
            /**
             * \brief Specialisation of TPGVertex method to accept only ActionEdge or ConnectionEdge
             *
             * \param[in] edge the TPGEdge pointer to be added to the outgoingEdges
             *                 Set.
             */
            virtual void addOutgoingEdge(TPG::TPGEdge* edge) override;

            
            /**
             * \brief Get a list to outgoing decision edges of this TPGVertex.
             */
            std::list<TPGEdge*> getOutgoingConnectionEdges() const;

            /**
             * \brief Get a list to outgoing action edges of this TPGVertex.
             */
            std::list<TPGEdge*> getOutgoingActionEdges() const;

            /**
             * \brief Return the action edge corresponding to the action class
             * 
             * Return a pointer pointing to the edge linked to the action class. The pointer is set to nullptr if the action is not founded 
             * 
             * \param[in] actionClass int corresponding to the action class searched.
             */
            TPG::TPGActionEdge* getEdgeOfAction(uint64_t actionClass) const;
        
            /**
             * \brief Update the assessed actions
             */
            virtual void updateAssessedActions();
    };

}; // namespace TPG

#endif
