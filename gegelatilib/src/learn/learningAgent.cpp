/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019 - 2024) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019 - 2022)
 * Nicolas Sourbier <nsourbie@insa-rennes.fr> (2019 - 2020)
 * Pierre-Yves Le Rolland-Raumer <plerolla@insa-rennes.fr> (2020)
 * Quentin Vacher <qvacher@insa-rennes.fr> (2023 - 2024)
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

#include "data/hash.h"
#include "learn/evaluationResult.h"
#include "mutator/rng.h"
#include "mutator/tpgMutator.h"
#include "tpg/tpgExecutionEngine.h"

#include "learn/learningAgent.h"

std::shared_ptr<TPG::TPGGraph> Learn::LearningAgent::getTPGGraph()
{
    return this->tpg;
}

const Archive& Learn::LearningAgent::getArchive() const
{
    return this->archive;
}

const Environment& Learn::LearningAgent::getEnvironment() const
{
    return this->env;
}

Mutator::RNG& Learn::LearningAgent::getRNG()
{
    return this->rng;
}

void Learn::LearningAgent::init(uint64_t seed)
{
    // Initialize Randomness
    this->rng.setSeed(seed);

    // Initialize the tpg
    Mutator::TPGMutator::initRandomTPG(
        *this->tpg, params.mutation, this->rng,
        this->learningEnvironment.getNbActions());

    // Clear the archive
    this->archive.clear();

    // Clear the best root
    this->bestAgent = {nullptr, nullptr};
}

void Learn::LearningAgent::addLogger(Log::LALogger& logger)
{
    logger.doValidation = this->params.doValidation;
    // logs for example the headers of the columns the logger will print
    loggers.push_back(std::reference_wrapper<Log::LALogger>(logger));
}

bool Learn::LearningAgent::isAgentEvalSkipped(
    const TPG::TPGAgent& agent,
    std::shared_ptr<Learn::EvaluationResult>& previousResult) const
{
    // Has the root already been evaluated more times than
    // params.maxNbEvaluationPerPolicy
    const auto& iter = this->resultsPerAgent.find(&agent);
    if (iter != this->resultsPerAgent.end()) {
        // The root has already been evaluated
        previousResult = iter->second;
        return iter->second->getNbEvaluation() >=
               this->params.maxNbEvaluationPerPolicy;
    }
    else {
        previousResult = nullptr;
        return false;
    }
}

std::shared_ptr<Learn::EvaluationResult> Learn::LearningAgent::evaluateJob(
    TPG::TPGExecutionEngine& tee, const Job& job, uint64_t generationNumber,
    Learn::LearningMode mode, LearningEnvironment& le) const
{
    // Only consider the first root of jobs as we are not in adversarial mode
    const TPG::TPGAgent* agent = job.getAgent();

    // Skip the root evaluation process if enough evaluations were already
    // performed. In the evaluation mode only.
    std::shared_ptr<Learn::EvaluationResult> previousEval;
    if (mode == LearningMode::TRAINING &&
        this->isAgentEvalSkipped(*agent, previousEval)) {
        return previousEval;
    }

    // Init results
    double result = 0.0;

    // Number of evaluations
    uint64_t nbEvaluation = (mode == LearningMode::TRAINING) ? this->params.nbIterationsPerPolicyEvaluation:this->params.nbIterationsPerPolicyValidation;

    // Evaluate nbIteration times
    for (auto iterationNumber = 0; iterationNumber < nbEvaluation; iterationNumber++) {
        // Compute a Hash
        Data::Hash<uint64_t> hasher;
        uint64_t hash = hasher(generationNumber) ^ hasher(iterationNumber);

        // Reset the learning Environment
        le.reset(hash, mode, iterationNumber, generationNumber);

        uint64_t nbActions = 0;
        while (!le.isTerminal() &&
               nbActions < this->params.maxNbActionsPerEval) {
            // Get the actions
            std::vector<double> actionsID =
                tee.executeFromRoot(*agent, le.getInitActions()).second;
            // Do it
            le.doActions(actionsID);
            // Count actions
            nbActions++;
        }

        // Update results
        result += le.getScore();
    }

    // Create the EvaluationResult
    auto evaluationResult =
        std::shared_ptr<EvaluationResult>(new EvaluationResult(
            result / (double)nbEvaluation,
            nbEvaluation));

    // Combine it with previous one if any
    if (previousEval != nullptr) {
        *evaluationResult += *previousEval;
    }
    return evaluationResult;
}

std::multimap<std::shared_ptr<Learn::EvaluationResult>, const TPG::TPGAgent*>
Learn::LearningAgent::evaluateAllRoots(uint64_t generationNumber,
                                       Learn::LearningMode mode)
{
    std::multimap<std::shared_ptr<EvaluationResult>, const TPG::TPGAgent*>
        result;

    // Create the TPGExecutionEngine for this evaluation.
    // The engine uses the Archive only in training mode.
    std::unique_ptr<TPG::TPGExecutionEngine> tee =
        this->tpg->getFactory().createTPGExecutionEngine(
            this->env,
            (mode == LearningMode::TRAINING) ? &this->archive : NULL);

    auto agents = tpg->getAgents();
    for (int i = 0; i < agents.size(); i++) {
        auto job = makeJob(agents.at(i), mode);
        this->archive.setRandomSeed(job->getArchiveSeed());
        std::shared_ptr<EvaluationResult> avgScore = this->evaluateJob(
            *tee, *job, generationNumber, mode, this->learningEnvironment);
        result.emplace(avgScore, (*job).getAgent());
    }

    return result;
}

std::shared_ptr<Learn::EvaluationResult> Learn::LearningAgent::evaluateOneAgent(
    uint64_t generationNumber, Learn::LearningMode mode,
    const TPG::TPGAgent* agent)
{
    // Retrieve the index of the root TPGVertex
    const std::vector<const TPG::TPGAgent*> agents = tpg->getAgents();
    std::vector<const TPG::TPGAgent*>::const_iterator iterator =
        std::find(agents.begin(), agents.end(), agent);
    if (iterator == agents.end()) {
        throw std::runtime_error("The vertex to evaluate does not exist in the "
                                 "TPGGraph of the LearningAgent.");
    }

    // Create the TPGExecutionEngine for this evaluation.
    // The engine uses the Archive only in training mode.
    std::unique_ptr<TPG::TPGExecutionEngine> tee =
        this->tpg->getFactory().createTPGExecutionEngine(
            this->env,
            (mode == LearningMode::TRAINING) ? &this->archive : NULL);

    // Create and evaluate the job
    auto job = makeJob(*iterator, mode);
    this->archive.setRandomSeed(job->getArchiveSeed());
    std::shared_ptr<EvaluationResult> avgScore = this->evaluateJob(
        *tee, *job, generationNumber, mode, this->learningEnvironment);

    // Return the result
    return avgScore;
}

void Learn::LearningAgent::trainOneGeneration(uint64_t generationNumber)
{
    for (auto logger : loggers) {
        logger.get().logNewGeneration(generationNumber);
    }

    // Populate Sequentially
    Mutator::TPGMutator::populateTPG(
        *this->tpg, this->archive, this->params.mutation, this->rng,
        generationNumber, maxNbThreads);
    for (auto logger : loggers) {
        logger.get().logAfterPopulateTPG();
    }

    // Evaluate
    auto results =
        this->evaluateAllRoots(generationNumber, LearningMode::TRAINING);
    for (auto logger : loggers) {
        logger.get().logAfterEvaluate(results);
    }

    // Save the best score of this generation
    this->updateBestScoreLastGen(results);

    // Remove worst performing roots
    decimateWorstRoots(results);
    // Update the best
    this->updateEvaluationRecords(results);

    for (auto logger : loggers) {
        logger.get().logAfterDecimate();
    }

    // Does a validation or not according to the parameter doValidation
    if (params.doValidation) {
        std::multimap<std::shared_ptr<Learn::EvaluationResult>, const TPG::TPGAgent*> validationResults;

        if(generationNumber % params.stepValidation == 0 || generationNumber == params.nbGenerations - 1){
            validationResults = evaluateAllRoots(generationNumber, Learn::LearningMode::VALIDATION);
        }
        for (auto logger : loggers) {
            logger.get().logAfterValidate(validationResults);
        }
    }

    for (auto logger : loggers) {
        logger.get().logEndOfTraining();
    }

    for(auto v: tpg->getRootVertices()){
        std::cout<<" "<<v->getProportionSpecies()<<" - ";
    }
    std::cout<<std::endl;

}

std::unordered_map<const TPG::TPGVertex*, double> Learn::LearningAgent::computeSoftmaxSpeciesScores(
    std::multimap<std::shared_ptr<EvaluationResult>, const TPG::TPGAgent*>&
    results)
{
    double averageScores = 0;
    double stdScores = 0;
    double expSumScores = 0;
    
    // Get the average score of each species
    std::unordered_map<const TPG::TPGVertex*, double> scoreSpecies;


    // Fill the map with root species.
    for(auto vertex: tpg->getRootVertices()){
        scoreSpecies.insert({vertex, 0.0});
    }
    // Compute the sum of results.
    for(auto& pair: results){
        scoreSpecies[pair.second->getRootSpecies()] += pair.first->getResult();
    }
    // Divide by the number of agents to the average, while getting the sum of the scores.
    for(auto& pair: scoreSpecies){
        pair.second /= tpg->getNbAgentsOfSpecies(*pair.first);

        averageScores += pair.second;
    }
    averageScores /= scoreSpecies.size();

    for(auto& pair: scoreSpecies){
        stdScores += std::pow(averageScores - pair.second, 2);
    }
    stdScores = std::sqrt(stdScores / scoreSpecies.size());

    //std::cout<<"\nAverage results with std and average : "<<stdScores<<" "<<averageScores<<std::endl;;
    // Standardize the scores and get the sum of exponential scores
    for (auto& pair: scoreSpecies){
        //std::cout<<pair.second<<" ";
        pair.second = (pair.second - stdScores) / averageScores;
        //std::cout<<pair.second<<" - ";
        expSumScores += std::exp(pair.second);
    }
    //std::cout<<"\nSoftmax results with expSum : "<<expSumScores<<std::endl;
    // Apply softmax on the scores
    for (auto& pair: scoreSpecies){
        pair.second = std::exp(pair.second) / expSumScores;
        //std::cout<<pair.second<<" ";
    }
    //std::cout<<std::endl;

    return scoreSpecies;
}

void Learn::LearningAgent::decimateWithTournament(
    std::multimap<std::shared_ptr<EvaluationResult>, const TPG::TPGAgent*>&
        results)
{



    auto scoreSpecies = computeSoftmaxSpeciesScores(results);

    bool deleteMin = scoreSpecies.size() == 8;
    double min = 1;
    const TPG::TPGVertex* minVertex;
    if(deleteMin){
        for(auto& pair: scoreSpecies){
            if(min > pair.second){
                min = pair.second;
                minVertex = pair.first;
            }
        }
    }


    for(auto& pair: scoreSpecies){

        if(pair.second < 0.1 || (deleteMin && pair.first == minVertex && false)){ // Don't forget to add an incremental value for letting a new species survive


            // Score of the species is to low, species is removed
            auto agents = tpg->getAgentsOfSpecies(*pair.first);
            while(agents.size() > 0){
                const TPG::TPGAgent* agent = agents.front();

                for(auto it = results.begin(); it != results.end(); ){
                    if((*it).second == agent){
                        it = results.erase(it);
                        break;
                    } else {
                        it++;
                    }
                }

                this->resultsPerAgent.erase(agent);
                tpg->removeAgent(*agent);
                agents.pop_front();


            }
            tpg->removeSpecies(*pair.first);
            tpg->removeVertex(*pair.first);
        }

    }

    // If the size is differnt, it mean some species have been deleted, a new softmax score need to be computed.
    if(scoreSpecies.size() != tpg->getNbRootVertices()){
        scoreSpecies = computeSoftmaxSpeciesScores(results);
    }
    std::vector<std::pair<const TPG::TPGVertex*, std::multimap<std::shared_ptr<EvaluationResult>, const TPG::TPGAgent*>>> resultsAllSpecies;
    for(const TPG::TPGVertex* species: tpg->getRootVertices()){

        // Set the proportion of the species.
        tpg->setProportionOfSpecies(*species, scoreSpecies.at(species));

        std::multimap<std::shared_ptr<EvaluationResult>, const TPG::TPGAgent*> resultsSpecies;
        for(auto& r : results) {
            if(r.second->getRootSpecies() == species) {
                resultsSpecies.insert(r);
            }
        }
        resultsAllSpecies.push_back({species, resultsSpecies});
    }

    for(auto& pair: resultsAllSpecies){

        const TPG::TPGVertex* species = pair.first;
        // Get the results of this species only
        std::multimap<std::shared_ptr<EvaluationResult>, const TPG::TPGAgent*>& resultsSpecies = pair.second;

        size_t nbAgentsInTournament = resultsSpecies.size() * params.ratioDeletedRoots;

        // Create subVector of results without the best agents.
        std::vector<std::pair<std::shared_ptr<EvaluationResult>, const TPG::TPGAgent*>> elements;
        auto it = resultsSpecies.begin();
        for (size_t i = 0; i < nbAgentsInTournament; ++i) {
            elements.push_back(*it++);
        }

            
        for (size_t i = 0; i < nbAgentsInTournament; i += params.sizeTournament) {
            std::multimap<std::shared_ptr<EvaluationResult>, const TPG::TPGAgent*> subMap;
            
            // Fill subMap with a size corresponding to the hardness of the tournament.
            for (size_t j = i; j < i + params.sizeTournament && j < nbAgentsInTournament; ++j) {

                uint64_t index = rng.getUnsignedInt64(0, elements.size() - 1);
                subMap.insert(elements[index]);
                elements.erase(elements.begin() + index);
            }

            // After the subMap is filled, erased the worse results from it, from the graph, and from the original results.
            while(subMap.size() != 1){
                tpg->removeAgent(*subMap.begin()->second);
                subMap.erase(subMap.begin());
                
            }
            
            tpg->setToBeDeleted(*subMap.begin()->second);
        }

        // Delete from results and resultsPerRoot
        auto itDel = resultsSpecies.begin();
        for (size_t i = 0; i < nbAgentsInTournament && itDel != resultsSpecies.end(); ++i) {
            this->resultsPerAgent.erase(itDel->second);
            for (auto it = results.begin(); it != results.end(); ) {
                if (it->second == itDel->second)
                    it = results.erase(it);
                else
                    ++it;
            }
            itDel++;
        }

        itDel = resultsSpecies.begin();
        while(resultsSpecies.size() > 0){
            itDel = resultsSpecies.erase(itDel);
        }
    }
}

void Learn::LearningAgent::decimateWorstRoots(
    std::multimap<std::shared_ptr<EvaluationResult>, const TPG::TPGAgent*>&
        results)
{

    if(params.useTournamentSelection){
        return decimateWithTournament(results);
    }



    // Estimate the number of expected roots to keep
    size_t nbExpectedRoots = floor(this->params.ratioDeletedRoots *
                                   (double)params.mutation.tpg.nbRoots);

    auto i = 0;
    while (i < nbExpectedRoots && results.size() > 0) {

        // If the root is an action, do not remove it in discrete environment!
        const TPG::TPGAgent* root = results.begin()->second;

        tpg->removeAgent(*results.begin()->second);
        // Removed stored result (if any)
        this->resultsPerAgent.erase(results.begin()->second);

        results.erase(results.begin());

        // Increment loop counter
        i++;
    }
}

uint64_t Learn::LearningAgent::train(volatile bool& altTraining,
                                     bool printProgressBar)
{
    const int barLength = 50;
    uint64_t generationNumber = 0;

    while (!altTraining && generationNumber < this->params.nbGenerations) {
        // Train one generation
        trainOneGeneration(generationNumber);
        generationNumber++;

        // Print progressBar (homemade, probably not ideal)
        if (printProgressBar) {
            printf("\rTraining ["); // back
            // filling ratio
            double ratio =
                (double)generationNumber / (double)this->params.nbGenerations;
            int filledPart = (int)((double)ratio * (double)barLength);
            // filled part
            for (int i = 0; i < filledPart; i++) {
                printf("%c", (char)219);
            }

            // empty part
            for (int i = filledPart; i < barLength; i++) {
                printf(" ");
            }

            printf("] %4.2f%%", ratio * 100.00);
        }
    }

    if (printProgressBar) {
        if (!altTraining) {
            printf("\nTraining completed\n");
        }
        else {
            printf("\nTraining alted at generation %" PRIu64 ".\n",
                   generationNumber);
        }
    }
    return generationNumber;
}

void Learn::LearningAgent::updateEvaluationRecords(
    const std::multimap<std::shared_ptr<EvaluationResult>,
                        const TPG::TPGAgent*>& results)
{
    { // Update resultsPerRoot
        for (auto result : results) {

            auto mapIterator = this->resultsPerAgent.find(result.second);
            if (mapIterator == this->resultsPerAgent.end()) {
                // First time this root is evaluated
                this->resultsPerAgent.emplace(result.second, result.first);
            }
            else if (result.first != mapIterator->second) {
                // This root has already been evaluated.
                // If the received result pointer is different from the one
                // stored in the map, update the one in the map by replacing it
                // with the new one (which was combined with the pre-existing
                // one in evalRoot)
                mapIterator->second = result.first;
                // If the received result is associated to the current bestRoot,
                // update it.
                if (result.second == this->bestAgent.first) {
                    this->bestAgent.second = result.first;
                }
            }
        }
    }

    { // Update bestRoot
        auto iterator = --results.end();
        const std::shared_ptr<EvaluationResult> evaluation = iterator->first;
        const TPG::TPGAgent* candidate = iterator->second;
        // Test the three replacement cases
        // from the simpler to the most complex to test

        // Replace the best root
        this->bestAgent = {candidate, evaluation};

        // Otherwise do nothing
    }
}

const std::pair<const TPG::TPGAgent*,
                std::shared_ptr<Learn::EvaluationResult>>&
Learn::LearningAgent::getBestAgent() const
{
    return this->bestAgent;
}

void Learn::LearningAgent::updateBestScoreLastGen(
    std::multimap<std::shared_ptr<Learn::EvaluationResult>,
                  const TPG::TPGAgent*>& results)
{
    auto iterator = --results.end();
    bestScoreLastGen = iterator->first->getResult();
}

double Learn::LearningAgent::getBestScoreLastGen() const
{
    return bestScoreLastGen;
}

void Learn::LearningAgent::keepBestPolicy()
{
    // Evaluate all roots
    if (this->tpg->hasAgent(*this->bestAgent.first)) {
        auto bestRootVertex = this->bestAgent.first;

        auto agents = this->tpg->getAgents();
        for (auto agent : agents) {
            if (agent != bestRootVertex) {
                tpg->removeAgent(*agent);
            }
        }
    }
}

std::shared_ptr<Learn::Job> Learn::LearningAgent::makeJob(
    const TPG::TPGAgent* agent, Learn::LearningMode mode, int idx,
    TPG::TPGGraph* tpgGraph)
{
    // sets the tpg to the Learning Agent's one if no one was specified
    tpgGraph = tpgGraph == nullptr ? tpg.get() : tpgGraph;

    // Before each root evaluation, set a new seed for the archive in
    // TRAINING Mode Else, archiving should be deactivate anyway
    uint64_t archiveSeed = 0;
    if (mode == LearningMode::TRAINING) {
        archiveSeed = this->rng.getUnsignedInt64(0, UINT64_MAX);
    }

    if (tpgGraph->getNbRootVertices() > 0) {
        return std::make_shared<Learn::Job>(
            Learn::Job(agent, archiveSeed, idx));
    }
    return nullptr;
}

std::queue<std::shared_ptr<Learn::Job>> Learn::LearningAgent::makeJobs(
    Learn::LearningMode mode, TPG::TPGGraph* tpgGraph)
{
    // sets the tpg to the Learning Agent's one if no one was specified
    tpgGraph = tpgGraph == nullptr ? tpg.get() : tpgGraph;

    std::queue<std::shared_ptr<Learn::Job>> jobs;
    auto agents = tpgGraph->getAgents();
    for (int i = 0; i < agents.size(); i++) {
        auto job = makeJob(agents.at(i), mode, i);
        jobs.push(job);
    }
    return jobs;
}

void Learn::LearningAgent::forgetPreviousResults()
{
    resultsPerAgent.clear();
    bestAgent.first = nullptr;
    bestAgent.second = nullptr;
}
