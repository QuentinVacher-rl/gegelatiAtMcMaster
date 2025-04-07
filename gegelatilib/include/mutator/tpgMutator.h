/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019 - 2024) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019 - 2022)
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

#ifndef TPG_MUTATOR_H
#define TPG_MUTATOR_H

#include <thread>

#include "archive.h"
#include "mutator/mutationParameters.h"
#include "tpg/tpgGraph.h"

namespace Mutator {
    namespace TPGMutator {
        /**
         * \brief Initialize a random TPGGraph.
         *
         * Following Stephen Kelly's PhD Thesis, the created TPGGraph will
         * contain:
         * - Exactly nbAction TPGAction vertices.
         * - Exactly nbAction TPGTeam vertices
         * - Exactly 2*nbAction Programs
         * - Between 2 and maxInitOutgoingEdges TPGEdge per TPGTeam, where
         *   - Each TPGEdge connects a TPGTeam with a TPGAction.
         *   - Each TPGTeam is connected to a TPGAction at most once.
         *   - Each TPGTeam is connected to at least 2 distinct TPGAction
         *   - Each Program is used at most once per TPGTeam.
         *   - Each Program always leads to the same TPGAction.
         *   - Each Program is approximately used the same number of time.
         * Hence, the maxInitOutgoingEdges value can not be greater than
         * nbAction.
         *
         * If the TPGGraph is not empty, all its vertices (and hence all its
         * edges) are removed before initialization.
         *
         * \param[in,out] graph the initialized TPGGraph.
         * \param[in] params the Parameters for the mutation.
         * \param[in] rng Random Number Generator used in the mutation process.
         * \param[in] nbAction number of actions that will be usable for
         * interacting with this LearningEnviromnent.
         * \throw std::runtime_error if maxInitOutgoingEdges exceeds nbAction.
         *        Or if nbAction is smaller than 1.
         */
        void initRandomTPG(TPG::TPGGraph& graph,
                           const MutationParameters& params, Mutator::RNG& rng,
                           uint64_t nbAction);


        #if 0           
        /**
         * \brief Select a random outgoingEdge of the given TPGVertex and removes
         * it from the TPGGraph.
         *
         * \param[in,out] graph the TPGGraph within which the team is stored.
         * \param[in] vertex the TPGVertex whose outgoingActionEdges will be altered.
         * \param[in] rng Random Number Generator used in the mutation process.
         */
        void removeRandomActionEdge(TPG::TPGGraph& graph, const TPG::TPGVertex& vertex,
                              Mutator::RNG& rng);

        /**
         * \brief Add a new outgoing TPGActionEdge to the TPGVertex within the TPGGraph.
         *
         *
         * \param[in,out] graph the TPGGraph within which the action is stored.
         * \param[in] vertex the TPGVertex whose outgoingActionEdges will be altered.
         * \param[in] preExistingEdges the TPGEdge candidates for cloning.
         * \param[in] rng Random Number Generator used in the mutation process.
         */
        void addRandomActionEdge(
            TPG::TPGGraph& graph, const TPG::TPGVertex& vertex,
            const std::list<const TPG::TPGEdge*>& preExistingActionEdges,
            Mutator::RNG& rng);


        /**
         * \brief Swap two edges of TPGVertex.
         *
         * \param[in,out] graph the TPGGraph within which the team and edge are
         *                stored.
         * \param[in] vertex the TPGVertex whose outgoingActionEdges will be altered.
         * \param[in] rng Random Number Generator used in the mutation process.
         */
        void swapActionEdges(            
            TPG::TPGGraph& graph, const TPG::TPGVertex& vertex, Mutator::RNG& rng);
            #endif
        /**
         * \brief Mutate the edge of an action Vertex
         *
         *
         * \param[in,out] graph the TPGGraph within which the team and edge are
         *                stored.
         * \param[in] vertex the TPGVertex whose actionEdges will be altered.
         * \param[in] edge the TPGActionEdge mutated
         * \param[in,out] newPrograms List of new Program created during
         *                mutations of the TPGTeam. The behavior of these
         *                Program must be mutated to complete the mutation
         *                process.
         * \param[in] params Probability parameters for the mutation.
         * \param[in] rng Random Number Generator used in the mutation process.
         */
        void mutateTPGEdge(
            TPG::TPGGraph& graph, const TPG::TPGVertex& vertex, TPG::TPGEdge* edge,
            std::list<std::shared_ptr<Program::Program>>& newPrograms,
            const Mutator::MutationParameters& params, Mutator::RNG& rng);


        /**
         * \brief Mutate a whole species with the exact same edge duplicated.
         *
         *
         * \param[in,out] graph the TPGGraph within which the team and edge are
         *                stored.
         * \param[in] species vector with all the agents of the species
         * \param[in,out] newPrograms List of new Program created during
         *                mutations of the TPGTeam. The behavior of these
         *                Program must be mutated to complete the mutation
         *                process.
         * \param[in] params Probability parameters for the mutation.
         * \param[in] rng Random Number Generator used in the mutation process.
         */
        void duplicateEdgeSpecies(TPG::TPGGraph& graph, 
            std::vector<const TPG::TPGVertex*> species,
            std::list<std::shared_ptr<Program::Program>>& newPrograms,
            const Mutator::MutationParameters& params, Mutator::RNG& rng);
        /**
         * \brief Mutate a whole species with the exact same edge deleted.
         *
         *
         * \param[in,out] graph the TPGGraph within which the team and edge are
         *                stored.
         * \param[in] species vector with all the agents of the species
         * \param[in,out] newPrograms List of new Program created during
         *                mutations of the TPGTeam. The behavior of these
         *                Program must be mutated to complete the mutation
         *                process.
         * \param[in] params Probability parameters for the mutation.
         * \param[in] rng Random Number Generator used in the mutation process.
         */
        void deleteEdgeSpecies(TPG::TPGGraph& graph, 
            std::vector<const TPG::TPGVertex*> species,
            std::list<std::shared_ptr<Program::Program>>& newPrograms,
            const Mutator::MutationParameters& params, Mutator::RNG& rng);
        /**
         * \brief Mutate a whole species with the exact same change of action.
         *
         *
         * \param[in,out] graph the TPGGraph within which the team and edge are
         *                stored.
         * \param[in] species vector with all the agents of the species
         * \param[in,out] newPrograms List of new Program created during
         *                mutations of the TPGTeam. The behavior of these
         *                Program must be mutated to complete the mutation
         *                process.
         * \param[in] params Probability parameters for the mutation.
         * \param[in] rng Random Number Generator used in the mutation process.
         */
        void changeActionClassSpecies(TPG::TPGGraph& graph, 
            std::vector<const TPG::TPGVertex*> species,
            std::list<std::shared_ptr<Program::Program>>& newPrograms,
            const Mutator::MutationParameters& params, Mutator::RNG& rng);
        /**
         * \brief Mutate a whole species with the exact extension of graph.
         *
         *
         * \param[in,out] graph the TPGGraph within which the team and edge are
         *                stored.
         * \param[in] species vector with all the agents of the species
         * \param[in,out] newPrograms List of new Program created during
         *                mutations of the TPGTeam. The behavior of these
         *                Program must be mutated to complete the mutation
         *                process.
         * \param[in] params Probability parameters for the mutation.
         * \param[in] rng Random Number Generator used in the mutation process.
         */
        void extendSpecies(TPG::TPGGraph& graph, 
            std::vector<const TPG::TPGVertex*> species,
            std::list<std::shared_ptr<Program::Program>>& newPrograms,
            const Mutator::MutationParameters& params, Mutator::RNG& rng);


        /**
         * \brief Mutate a whole species with the exact same graph mutation.
         *
         *
         * \param[in,out] graph the TPGGraph within which the team and edge are
         *                stored.
         * \param[in] species vector with all the agents of the species
         * \param[in,out] newPrograms List of new Program created during
         *                mutations of the TPGTeam. The behavior of these
         *                Program must be mutated to complete the mutation
         *                process.
         * \param[in] params Probability parameters for the mutation.
         * \param[in] rng Random Number Generator used in the mutation process.
         */
        void mutateSpecies(TPG::TPGGraph& graph, 
            std::vector<const TPG::TPGVertex*> species,
            std::list<std::shared_ptr<Program::Program>>& newPrograms,
            const Mutator::MutationParameters& params, Mutator::RNG& rng);

        /**
         * \brief Copy and mutate a TPGAction vertex 
         *
         * This function take a TPGAction, copy it and mutate it.
         * It can randomly change one of the ActionEdges by another ActionEdges of another TPGAction, however it must be the same index of actionEdges.
         * If not, it can randomly swap two actionEdges.
         * If not, it can mutate the program on the actionEdges.
         *
         * \param[in,out] graph the TPGGraph within which the team and edge are
         *                stored.
         * \param[in] vertex the TPGVertex whose actionEdges will be altered.
         * \param[in] team the TPGTeam source of TPGAction
         * \param[in,out] newPrograms List of new Program created during
         *                mutations of the TPGTeam. The behavior of these
         *                Program must be mutated to complete the mutation
         *                process.
         * \param[in] params Probability parameters for the mutation.
         * \param[in] rng Random Number Generator used in the mutation process.
         */
        void mutateTPGVertex(
            TPG::TPGGraph& graph, const TPG::TPGVertex& vertex,
            std::list<std::shared_ptr<Program::Program>>& newPrograms,
            const Mutator::MutationParameters& params, Mutator::RNG& rng);




        /**
         * \brief Mutate the behavior of a Program and ensure its unicity
         * against the given Archive.
         *
         * \param[in,out] newProg Program whose behavior is being mutated.
         * \param[in] params Probability parameters for the mutation.
         * \param[in] archive Archive used to assess the uniqueness of the
         *            mutated Program behavior.
         * \param[in] rng Random Number Generator used in the mutation process.
         */
        void mutateProgramBehaviorAgainstArchive(
            std::shared_ptr<Program::Program>& newProg,
            const Mutator::MutationParameters& params, const Archive& archive,
            Mutator::RNG& rng);

        /**
         * \brief Function mutating the behavior of the given list of Program.
         *
         * \param[in] maxNbThreads Integer parameter controlling the number of
         *           threads used for parallel execution.Possible values are :
         *           -`0`and `1`: Do not use parallelism.
         *           -`n > 1`: Set the number of threads explicitly.
         * \param[in] newPrograms List of new Program to mutate.
         * \param[in] rng Random Number Generator used in the mutation process.
         * \param[in] params Probability parameters for the mutation.
         * \param[in] archive Archive used to assess the uniqueness of the
         * mutated Program behavior.
         */
        void mutateNewProgramBehaviors(
            const uint64_t& maxNbThreads,
            std::list<std::shared_ptr<Program::Program>>& newPrograms,
            Mutator::RNG& rng, const Mutator::MutationParameters& params,
            const Archive& archive);


        /**
         * \brief do a crossover by creating a new program for each child, with a block from each parent.
         * 
         * \param[in] graph current Graph
         * \param[in] childs new vertices
         * \param[in] edges TPGEdges copied
         * \param[in] params Probability parameters for the mutation.
         * \param[in] rng Random Number Generator used in the mutation process.
         * 
         */
        void crossProgram(
            TPG::TPGGraph& graph,
            std::vector<const TPG::TPGVertex*>& childs,
            std::vector<TPG::TPGEdge*>& edges,
            const Mutator::MutationParameters& params,
            Mutator::RNG& rng);

        /**
         * \brief do a crossover on the edges, without creating new programs
         * 
         * \param[in] graph current Graph
         * \param[in] childs new vertices
         * \param[in] edges TPGEdges copied
         * \param[in] params Probability parameters for the mutation.
         * \param[in] rng Random Number Generator used in the mutation process.
         * 
         */
        void crossEdges(
            TPG::TPGGraph& graph,
            std::vector<const TPG::TPGVertex*>& childs,
            std::vector<TPG::TPGEdge*>& edges,
            const Mutator::MutationParameters& params,
            Mutator::RNG& rng);

        /**
         * \brief do a crossover to create two new agents
         * 
         * \param[in] graph current Graph
         * \param[in] childs new vertices
         * \param[in] params Probability parameters for the mutation.
         * \param[in] rng Random Number Generator used in the mutation process.
         * 
         */
        void crossTPGVertices(
            TPG::TPGGraph& graph,
            std::vector<const TPG::TPGVertex*>& childs,
            const Mutator::MutationParameters& params,
            Mutator::RNG& rng);

        /**
         * \brief Create new root TPGTeam within the TPGGraph.
         *
         * This function create and add new root TPGTeam to the TPGGraph
         * until the targetted number of roots is reached. To create new root
         * TPGTeam, the function uses mutation operators on duplicates of
         * existing root TPGTeams of the TPGGraph.
         *
         * A few special cases are handled:
         * If the set of root vertices of the TPGGraph contains any TPGAction,
         * this TPGAction is ignored when selecting a candidate for duplication.
         * If the TPGGraph does not have any root TPGTeam, it is reinitialized
         * entirely with the initRandomTPG function.
         * If the given TPGGraph already has more root TPGVertex than the
         * targetted number of root teams, nothing happens.
         *
         * \param[in,out] graph the TPGGraph to mutate.
         * \param[in] archive Archive used to assess the uniqueness of the
         *            mutated Program behavior.
         * \param[in] params Probability parameters for the mutation.
         * \param[in] rng Random Number Generator used in the mutation process.
         * \param[in] nbAction number of actions that will be usable for
         * interacting with this LearningEnviromnent.
         * \param[in] maxNbThreads Integer parameter controlling the number of
         * threads used for parallel execution. Possible values are:
         *   - default:  Let the runtime decide using
         *               std::thread::hardware_concurrency().
         *   - `0` and `1`: Do not use parallelism.
         *   - `n > 1`: Set the number of threads explicitly.
         */
        void populateTPG(
            TPG::TPGGraph& graph, const Archive& archive,
            const Mutator::MutationParameters& params, Mutator::RNG& rng,
            uint64_t nbActions,
            uint64_t maxNbThreads = std::thread::hardware_concurrency());
    }; // namespace TPGMutator
};     // namespace Mutator

#endif
