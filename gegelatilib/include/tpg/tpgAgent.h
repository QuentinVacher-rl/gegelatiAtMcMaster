
#ifndef TPG_AGENT_H
#define TPG_AGENT_H

#include <tpg/tpgVertex.h>
#include <tpg/tpgEdge.h>

namespace TPG {
    /**
     * \brief Class for storing a TPG Agent.
     */
    class TPGAgent
    {
        private: 

            /**
             * \brief map linking TPGEdges to a corresponding program
             */
            std::unordered_map<const TPG::TPGEdge*, std::shared_ptr<Program::Program>> programs; 

            /**
             * \brief root species of the agent
             */
            const TPG::TPGVertex* rootSpecies;

            /// True if the vertex should be deleted during evolution process
            bool toBeDeleted = false;

        public:

        /**
         * Main constructor of a TPGAgent.
         * 
         * \param[in] rootSp root that indicate the species of the agent
         */
        TPGAgent(const TPG::TPGVertex* rootSp): rootSpecies{rootSp} {};

        /**
         * \brief Get a const reference to the Program of the TPGAgent.
         *
         * \param[in] edge TPGEdge linked to the program
         * 
         * \return a const reference to the Program of the TPGAgent linked to the TPGEdge.
         */
        virtual Program::Program& getProgram(const TPG::TPGEdge& edge) const;

        /**
         * \brief Set a new Program for the TPGAgent with the corresponding edge in key.
         *
         * \param[in] prog the new shared pointer to a Program.
         * \param[in] edge the new edge to the program
         */
        virtual void setProgram(const TPG::TPGEdge* edge, std::shared_ptr<Program::Program> prog);

        /**
         * \brief Get the shared_pointer to the Program.
         *
         * This method is voluntarily non-const to make sure that only the
         * TPGGraph containing the edge can use it.
         *
         * \param[in] edge TPGEdge linked to the program
         * \return a copy of the program attribute.
         */
        virtual std::shared_ptr<Program::Program> getProgramSharedPointer(const TPG::TPGEdge* edge) const;

        /**
         * \brief erase a pair of edge/program to the programs map.
         * 
         * \param[in] edge the edge to be erased.
         */
        virtual bool deletePair(const TPG::TPGEdge* edge);

        /**
         * \brief return the unordered_map of TPGEdge-Program of the TPGagent.
         */
        virtual const std::unordered_map<const TPG::TPGEdge*, std::shared_ptr<Program::Program>>& getPrograms() const ;

        /**
         * \brief return a const pointer to the TPGVertex* at the root of the species of the TPGAgent.
         */
        virtual const TPG::TPGVertex* getRootSpecies() const;


        /**
         * \brief set a new root species for this agent
         * 
         * \param newRoot new root species set.
         */
        virtual void setRootSpecies(const TPG::TPGVertex* newRoot);

        /**
         * \brief method that return the size of the programs map.
         */
        virtual size_t agentSize() const;

        /**
         * Set if the vertex should be deleted during evolution process. 
         * 
         * \param[in] status boolean to indicate if the vertex should be deleted.
         */
        virtual void setToBeDeleted(bool status);

        /**
         * Return if the vertex should be deleted during evolution process. 
         */
        virtual bool isToBeDeleted() const; 
    };
};

#endif 