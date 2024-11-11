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


#include <inttypes.h>
#include <queue>


#include "learn/parallelEvoStratLearningAgent.h"



std::multimap<std::shared_ptr<Learn::EvaluationResult>, const std::map<Program::Program*, std::vector<double>>*>
    Learn::ParallelEvoStratLearningAgent::evaluateAllErrorWeights(uint64_t generationNumber,
                                       Learn::LearningMode mode)
{
    std::multimap<std::shared_ptr<Learn::EvaluationResult>, const std::map<Program::Program*, std::vector<double>>*>
        results;

    if (this->maxNbThreads <= 1 || !this->learningEnvironment.isCopyable())
    {
        results = Learn::EvoStratLearningAgent::evaluateAllErrorWeights(generationNumber, mode);
    } else {
        // Parallel mode
        this->evaluateAllErrorWeightsInParallel(generationNumber, mode, results);
    }
    return results;
}

void Learn::ParallelEvoStratLearningAgent::evaluateAllErrorWeightsInParallel(
    uint64_t generationNumber, LearningMode mode,
    std::multimap<std::shared_ptr<EvaluationResult>, const std::map<Program::Program*, 
                    std::vector<double>>*>& results)
{
    std::map<uint64_t,
             std::pair<std::shared_ptr<EvaluationResult>, std::shared_ptr<Job>>>
        resultsPerJobMap;

    evaluateAllErrorWeightsInParallelExecute(generationNumber, mode, resultsPerJobMap);

    evaluateAllErrorWeightsInParallelCompileResults(resultsPerJobMap, results);
}

void Learn::ParallelEvoStratLearningAgent::evaluateAllErrorWeightsInParallelExecute(
            uint64_t generationNumber, LearningMode mode,
            std::map<uint64_t, std::pair<std::shared_ptr<EvaluationResult>,
                                         std::shared_ptr<Job>>>&
                resultsPerJobMap
                )
{
    // Create and fill the queue for distributing work among threads
    // each root is associated to its number in the list for enabling the
    // determinism of stochastic archive storage.
    auto jobsToProcess = makeJobs(mode);

    // Create mutexes
    std::mutex rootsToProcessMutex;
    std::mutex resultsPerRootMutex;

    // Create the threads
    std::vector<std::thread> threads;
    for (auto i = 0; i < (this->maxNbThreads - 1); i++) {
        threads.emplace_back(std::thread(
            &ParallelEvoStratLearningAgent::slaveEvalJobThread, this, generationNumber,
            mode, std::ref(jobsToProcess), std::ref(rootsToProcessMutex),
            std::ref(resultsPerJobMap), std::ref(resultsPerRootMutex), false));
    }

    // Work in the main thread also, using the main environment
    this->slaveEvalJobThread(generationNumber, mode, jobsToProcess,
                             rootsToProcessMutex, resultsPerJobMap,
                             resultsPerRootMutex, true);

    // Join the threads
    for (auto& thread : threads) {
        thread.join();
    }

}

void Learn::ParallelEvoStratLearningAgent::slaveEvalJobThread(
    uint64_t generationNumber, Learn::LearningMode mode,
    std::queue<std::shared_ptr<Learn::Job>>& jobsToProcess,
    std::mutex& rootsToProcessMutex,
    std::map<uint64_t, std::pair<std::shared_ptr<EvaluationResult>,
                                 std::shared_ptr<Job>>>& resultsPerRootMap,
    std::mutex& resultsPerRootMapMutex,
    bool useMainEnvironment)
{

    // Clone learningEnvironment
    LearningEnvironment* privateLearningEnvironment =
        useMainEnvironment ? &this->learningEnvironment
                           : this->learningEnvironment.clone();

    // Create a TPGExecutionEngine
    Environment privateEnv(
        this->env.getInstructionSet(), this->env.getParams(), 
        privateLearningEnvironment->getDataSources(), this->env.getNbContinuousActions()
    );
    
    std::unique_ptr<TPG::TPGExecutionEngine> tee =
        this->tpg->getFactory().createTPGExecutionEngine(privateEnv, NULL);

    int i = 0;
    // Pop a job
    while (!jobsToProcess.empty()) { // Thread safe access to size
        i++;
        bool doProcess = false;
        std::shared_ptr<Learn::Job> jobToProcess;
        { // Mutuel exclusion zone
            std::lock_guard<std::mutex> lock(rootsToProcessMutex);
            if (!jobsToProcess.empty()) { // Additional verification after lock
                jobToProcess = jobsToProcess.front();
                jobsToProcess.pop();
                doProcess = true;
            }
        } // End of mutual exclusion zone

        // Processing to do?
        if (doProcess) {
            doProcess = false;

            std::shared_ptr<EvaluationResult> avgScore =
                this->evaluateJob(*tee, *jobToProcess, generationNumber, mode,
                                  *privateLearningEnvironment);

            { // Store result Mutual exclusion zone
                std::lock_guard<std::mutex> lock(resultsPerRootMapMutex);
                resultsPerRootMap.emplace(
                    jobToProcess->getIdx(),
                    std::make_pair(avgScore, jobToProcess));
            }
        }
    }

    // Clean up
    if (!useMainEnvironment) {
        delete privateLearningEnvironment;
    }
}


void Learn::ParallelEvoStratLearningAgent::evaluateAllErrorWeightsInParallelCompileResults(
    std::map<uint64_t, std::pair<std::shared_ptr<EvaluationResult>,
                                 std::shared_ptr<Job>>>& resultsPerJobMap,
    std::multimap<std::shared_ptr<EvaluationResult>, const std::map<Program::Program*, 
                    std::vector<double>>*>& results)
{
    // Merge the results
    for (auto& resultPerRoot : resultsPerJobMap) {
        if((*resultPerRoot.second.second).getErrorWeights() != nullptr){
            results.emplace(resultPerRoot.second.first,
                            (*resultPerRoot.second.second).getErrorWeights());
        } else {
            throw std::runtime_error("errorWeights attribute of the jobs should not be a null pointor");
        }
    }
}

