/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019 - 2022) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019 - 2022)
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

 #ifndef TPG_CONNECTION_EDGE_H
 #define TPG_CONNECTION_EDGE_H
 
#include "tpg/tpgEdge.h"
 
namespace TPG {
    // Declare class to make it usable as an attribute.
    class TPGVertex;

    /**
     * \brief Class representing connection edges of the Tangled Program Graphs, edge than connect vertecies with no program/bid system.
     */
    class TPGConnectionEdge : public TPGEdge
    {
    public:
        /// Default virtual destructor (for polymorphism)
        virtual ~TPGConnectionEdge() = default;

        /**
         * \brief Main constructor of the TPGEdge class.
         *
         * This constructor does not register the created TPGEdge in the
         * list of incoming or outgoing edges of the given TPGVertex.
         *
         * \param[in] src pointer to the source TPGVertex of the edge.
         * \param[in] dest pointer to the destination TPGVertex of the edge.
         */
        TPGConnectionEdge(const TPGVertex* src, const TPGVertex* dest)
                    : TPGEdge{src, dest, nullptr} {};

        /**
         * \brief Get a const reference to the Program of the TPGEdge.
         *
         * \return a const reference to the Program of the TPGEdge.
         */
        virtual Program::Program& getProgram() const override;

        /**
         * \brief Set a new Program for the TPGEdge.
         *
         * This method is const to enable use outside of the TPGGraph which is
         * the only class accessing the non-const TPGEdge. Since the program
         * pointer attribute is mutable, this method can successfully be used to
         * alter the program.
         *
         * \param[in] prog the new shared pointer to a Program.
         */
        virtual void setProgram(const std::shared_ptr<Program::Program> prog) const override;

        /**
         * \brief Get the shared_pointer to the Program.
         *
         * This method is voluntarily non-const to make sure that only the
         * TPGGraph containing the edge can use it.
         *
         * \return a copy of the program attribute.
         */
        virtual std::shared_ptr<Program::Program> getProgramSharedPointer() override;


        /// Delete the default constructor.
        TPGConnectionEdge() = delete;
    };
}; // namespace TPG

#endif
 