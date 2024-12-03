/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019 - 2023) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019 - 2022)
 * Nicolas Sourbier <nsourbie@insa-rennes.fr> (2019 - 2020)
 * Pierre-Yves Le Rolland-Raumer <plerolla@insa-rennes.fr> (2020)
 * Quentin Vacher <qvacher@insa-rennes.fr> (2023)
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

#ifndef PARALLEL_EVO_STRAT_LEARNING_AGENT_H
#define PARALLEL_EVO_STRAT_LEARNING_AGENT_H

#include "learn/parallelLearningAgent.h"
#include "learn/evoStratLearningAgent.h"

namespace Learn {

    /**
     * \brief Class used to control the learning steps of a TPGGraph within
     * a given LearningEnvironment.
     */
    class ParallelEvoStratLearningAgent : public EvoStratLearningAgent

    {
      protected:


        /**
         * \brief Method for evaluating all roots with parallelism.
         *
         * The work is delegated in two distinct methods (this structure is
         * made for inheritance purpose) : evaluateAllErrorWeightsInParallelExecute and
         * evaluateAllErrorWeightsInParallelCompileResults.
         *
         * \param[in] generationNumber the integer number of the current
         * generation. \param[in] mode the LearningMode to use during the policy
         * evaluation. \param[in] results Map to store the resulting score of
         * evaluated roots.
         */
        virtual void evaluateAllErrorWeightsInParallel(
            uint64_t generationNumber, LearningMode mode,
            std::multimap<std::shared_ptr<EvaluationResult>, const std::map<Program::Line*, 
                          std::vector<double>>*>& results);
        /**
         * \brief Subfunction of LearningMode which handles the
         * creation of threads, their execution and junction.
         *
         * @param[in] generationNumber the integer number of the current
         * generation.
         * @param[in] mode the LearningMode to use during the policy
         * evaluation.
         * @param[out] resultsPerJobMap map linking the job number with its
         * results and itself.
         */
        virtual void evaluateAllErrorWeightsInParallelExecute(
            uint64_t generationNumber, LearningMode mode,
            std::map<uint64_t, std::pair<std::shared_ptr<EvaluationResult>,
                                         std::shared_ptr<Job>>>&
                resultsPerJobMap
                );

        /**
         * \brief Function implementing the behavior of slave threads during
         * parallel evaluation of roots.
         *
         * \param[in] generationNumber the integer number of the current
         * generation. \param[in] mode the LearningMode to use during the policy
         * evaluation. \param[in,out] jobsToProcess Ordered list of jobs of
         * TPGVertex to process, stored as a pair with an id filling the
         * archiveMap. The jobs are groups of roots that shall be agents in the
         * same simulation, there is only 1 root if there is no adversarial
         * (e.g. if the environmnent is not multiplayer).
         * \param[in] rootsToProcessMutex Mutex protecting the
         * rootsToProcess \param[in] resultsPerRootMap Map to store the
         * resulting score of evaluated roots. \param[in] resultsPerRootMapMutex
         * Mutex protecting the results.
         * \param[in] useMainEnvironment Boolean that is true if we use the
         * declared LearningEnvironment, otherwise the method will clone it.
         */
        void slaveEvalJobThread(
            uint64_t generationNumber, LearningMode mode,
            std::queue<std::shared_ptr<Learn::Job>>& jobsToProcess,
            std::mutex& rootsToProcessMutex,
            std::map<uint64_t, std::pair<std::shared_ptr<EvaluationResult>,
                                         std::shared_ptr<Job>>>&
                resultsPerRootMap,
            std::mutex& resultsPerRootMapMutex, bool useMainEnvironment);

        /**
         * \brief Subfunction of evaluateAllRootsInParallel which handles the
         * gathering of results and the merge of the archives.
         *
         * This method just emplaces results from resultsPerJobMap, as each
         * job only contains 1 root is is quite easy.
         * The archive is merged with the mergeArchiveMap method.
         *
         * @param[in] resultsPerJobMap map linking the job number with its
         * results and itself.
         * @param[out] results map linking single results to their root vertex.
         */
        virtual void evaluateAllErrorWeightsInParallelCompileResults(
            std::map<uint64_t, std::pair<std::shared_ptr<EvaluationResult>,
                                         std::shared_ptr<Job>>>&
                resultsPerJobMap,
            std::multimap<std::shared_ptr<EvaluationResult>, const std::map<Program::Line*, 
                          std::vector<double>>*>& results);

      public:

        /**
         * \brief Constructor for EvoStratLearningAgent.
         *
         * EvoStratLearningAgent must be created with an existing learningAgent.
         *
         * \param[in] la The LearningAgent used.
         */
        ParallelEvoStratLearningAgent(ParallelLearningAgent& la, double sigma)
            : EvoStratLearningAgent(la, sigma){
                // overriding the maxNbThreads that basic evoLA defined to 1
                maxNbThreads = params.nbThreads;
            };




        /**
         * \brief Override on the evaluateAllRoots of LearningAgent
         *
         * \param[in] generationNumber the integer number of the current
         * generation.
         * \param[in] mode the LearningMode to use during the policy
         * evaluation.
         */
        virtual std::multimap<std::shared_ptr<EvaluationResult>, const std::map<Program::Line*, std::vector<double>>*>
        evaluateAllErrorWeights(uint64_t generationNumber, LearningMode mode) override;



    };
}; // namespace Learn

#endif
