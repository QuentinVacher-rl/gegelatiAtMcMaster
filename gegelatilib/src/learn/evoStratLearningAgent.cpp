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

#include "data/hash.h"
#include "learn/evaluationResult.h"
#include "mutator/rng.h"
#include "mutator/tpgMutator.h"
#include "tpg/tpgExecutionEngine.h"

#include "learn/evoStratlearningAgent.h"


void Learn::EvoStratLearningAgent::trainOneGeneration(uint64_t generationNumber)
{

    // For now, the number of roots should be equal to one to use this class.
    if(tpg->getNbRootVertices() != 1){
        throw std::runtime_error("Evolution Strategies is only available for one root, for now.");
    }


    for (auto logger : loggers) {
        logger.get().logNewGeneration(generationNumber);
    }

    // Generate some weights
    this->generateErrorWeights();
    
    // Evaluate
    auto results =
        this->evaluateAllErrorWeights(generationNumber, LearningMode::TRAINING);

    std::multimap<std::shared_ptr<Learn::EvaluationResult>, const TPG::TPGVertex *> fakeResultsForLogs;
    for(auto r: results){
        fakeResultsForLogs.insert(std::make_pair(r.first, this->tpg->getRootVertices().at(0)));
    }
    for (auto logger : loggers) {
        logger.get().logAfterEvaluate(fakeResultsForLogs);
    }

    // Learn
    this->doEvolutionStrategy(results);

    // Does a validation or not according to the parameter doValidation
    // We should always do one
    if (params.doValidation) {
        auto validationResults =
            evaluateAllRoots(generationNumber, Learn::LearningMode::VALIDATION);
        for (auto logger : loggers) {
            logger.get().logAfterValidate(validationResults);
        }
    }

    for (auto logger : loggers) {
        logger.get().logEndOfTraining();
    }
}

void Learn::EvoStratLearningAgent::generateErrorWeights()
{
        
    errorWeightsPopulation.clear();

    
    for (auto i = 0; i< nbAgents ; i++){
        errorWeightsPopulation.push_back(Mutator::TPGMutator::generateErrorWeights(
            *this->tpg, this->params.mutation, this->rng
        ));

        if(twinError){
            errorWeightsPopulation.push_back(Mutator::TPGMutator::generateTwinNegErrorWeights(
                *this->tpg, errorWeightsPopulation.back()
            ));
            i++;
        }
    }
}


void Learn::EvoStratLearningAgent::doEvolutionStrategy(
    std::multimap<std::shared_ptr<EvaluationResult>, 
                  const std::map<Program::Line*, std::vector<double>>*> results)

{

    for (auto &lines : *results.begin()->second) {
        Program::Line *line = lines.first;

        size_t nbUsed = 0;
        

        std::vector<double> originConstants;
        for(auto i=0; i<line->getNbConstants(); i++){
            double* constant = (double*)(line->cGetConstantHandler().getDataAt(typeid(Data::Constant), i).getSharedPointer<Data::Constant>().get());


            originConstants.push_back(*constant);
        }
        std::vector<double> evaluationWeights(line->getNbConstants());

        // Browse the results
        //std::cout<<"for one line"<<std::endl;
        std::vector<double> scores;
        for (const auto &resultEntry : results) {
            scores.push_back(resultEntry.first->getResult());


        }
        
        // Get the indices of scores
        std::vector<size_t> indices(scores.size());
        for (size_t i = 0; i < scores.size(); ++i) {
            indices[i] = i;
        }
        // Trier les indices en fonction des valeurs correspondantes dans vect
        std::sort(indices.begin(), indices.end(),
                [&scores](size_t a, size_t b) { return scores[a] < scores[b]; });

        // Créer un vecteur de rangs (même taille que vect)
        std::vector<double> ranks(scores.size());

        // Assigner les rangs en fonction des indices triés
        for (size_t rank = 0; rank < indices.size(); ++rank) {
            ranks[indices[rank]] = static_cast<double>(rank) / (indices.size() - 1) - 0.5;
        }

        uint64_t j = 0;
        for (const auto &resultEntry : results) {


            auto usedLines = resultEntry.first->getUsedLines();
            if(usedLines.find(line) != usedLines.end() ||true){
                // Get the score result
                double result = ranks[indices[j]];

                // Get the error weights
                const std::vector<double> &errorWeights = resultEntry.second->at(line);

                // Afficher ou utiliser les valeurs associées à ce programme pour ce résultat
                uint64_t i = 0;
                for (double value : errorWeights) {
                    evaluationWeights.at(i) += value * result;
                    i++;
                }


                nbUsed++;
            }


            j++;
        }

        if (nbUsed > 0) {
            for(size_t i = 0; i < line->getNbConstants(); i++){
                double newConstantsValue = originConstants.at(i) + lr * evaluationWeights.at(i) /( (double)nbAgents * sigma);
                line->getConstantHandler().setDataAt(typeid(Data::Constant), i, {static_cast<double>(newConstantsValue)});

            }
        }

    }
}


std::shared_ptr<Learn::EvaluationResult> Learn::EvoStratLearningAgent::evaluateJob(
    TPG::TPGExecutionEngine& tee, const Job& job, uint64_t generationNumber,
    Learn::LearningMode mode, LearningEnvironment& le) const
{

    if(mode == Learn::LearningMode::TRAINING){
        tee.setErrorWeights(job.getErrorWeights());
        tee.clearUsageLines();
    }

    std::shared_ptr<Learn::EvaluationResult> evaluationResult = LearningAgent::evaluateJob(
        tee, job, generationNumber, mode, le
    );

    if(mode == Learn::LearningMode::TRAINING){
        evaluationResult->addUsageLines(tee.getUsageLInes());
    }


    return evaluationResult;
}

std::multimap<std::shared_ptr<Learn::EvaluationResult>, const std::map<Program::Line*, std::vector<double>>*>
Learn::EvoStratLearningAgent::evaluateAllErrorWeights(uint64_t generationNumber,
                                       Learn::LearningMode mode)
{
    std::multimap<std::shared_ptr<EvaluationResult>, const std::map<Program::Line*, std::vector<double>>*>
        result;

    // Create the TPGExecutionEngine for this evaluation.
    // The engine uses the Archive only in training mode.
    std::unique_ptr<TPG::TPGExecutionEngine> tee =
        this->tpg->getFactory().createTPGExecutionEngine(
            this->env,
            (mode == LearningMode::TRAINING) ? &this->archive : NULL);

    auto roots = tpg->getRootVertices();
    for (int i = 0; i < errorWeightsPopulation.size(); i++) {
        auto job = makeJob(roots.at(0), mode, i);
        this->archive.setRandomSeed(job->getArchiveSeed());
        forgetPreviousResults();
        std::shared_ptr<EvaluationResult> avgScore = this->evaluateJob(
            *tee, *job, generationNumber, mode, this->learningEnvironment);
        result.emplace(avgScore, &errorWeightsPopulation.at(i));
    }

    return result;
}


std::queue<std::shared_ptr<Learn::Job>> Learn::EvoStratLearningAgent::makeJobs(
    Learn::LearningMode mode, TPG::TPGGraph* tpgGraph)
{
    // sets the tpg to the Learning Agent's one if no one was specified
    tpgGraph = tpgGraph == nullptr ? tpg.get() : tpgGraph;

    std::queue<std::shared_ptr<Learn::Job>> jobs;
    auto roots = tpgGraph->getRootVertices();
    for (int i = 0; i < errorWeightsPopulation.size(); i++) {
        auto job = makeJob(roots.at(0), mode, i);
        jobs.push(job);
    }
    return jobs;
}


std::shared_ptr<Learn::Job> Learn::EvoStratLearningAgent::makeJob(
    const TPG::TPGVertex* vertex, Learn::LearningMode mode, int idx,
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
        if(mode == LearningMode::TRAINING){
            return std::make_shared<Learn::Job>(
                Learn::Job({vertex}, archiveSeed, idx, &errorWeightsPopulation.at(idx)));
        }
        if(mode == LearningMode::VALIDATION){
            return std::make_shared<Learn::Job>(
                Learn::Job({vertex}, archiveSeed, idx));
        }
    }

    return nullptr;
}

bool Learn::EvoStratLearningAgent::isRootEvalSkipped(
    const TPG::TPGVertex& root,
    std::shared_ptr<Learn::EvaluationResult>& previousResult) const
{
    // Never skip a root evaluation during evo strat
    return false;
}

void Learn::EvoStratLearningAgent::decimateWorstRoots(
    std::multimap<std::shared_ptr<EvaluationResult>,
                    const TPG::TPGVertex*>& results) 
{
    std::cout<<"We should not be decimating roots in evoStrat Learning Agent"<<std::endl;
    return;
}
void Learn::EvoStratLearningAgent::updateEvaluationRecords(
    const std::multimap<std::shared_ptr<EvaluationResult>,
                        const TPG::TPGVertex*>& results)
{
    return;
}
