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
    
    falseTraining = false;
    // Evaluate
    auto results =
        this->evaluateAllErrorWeights(generationNumber, LearningMode::TRAINING);

    std::multimap<std::shared_ptr<Learn::EvaluationResult>, const TPG::TPGVertex *> fakeResultsForLogs;
    for(auto r: results){
        fakeResultsForLogs.insert(std::make_pair(r.first, this->tpg->getRootVertices().at(0)));
    }

    // Learn
    this->doEvolutionStrategy(results);

    for (auto logger : loggers) {
        dynamic_cast<Log::ESBasicLogger*>(&logger.get())->setSigma(sigma);
        logger.get().logAfterEvaluate(fakeResultsForLogs);
    }


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

    auto mutatedLines = Mutator::TPGMutator::selectMutatedLines(*this->tpg, this->params.mutation, this->rng);
    
    for (auto i = 0; i< nbAgents ; i++){
        errorWeightsPopulation.push_back(Mutator::TPGMutator::generateErrorWeights(
            mutatedLines, this->params.mutation, this->rng
        ));

        if(twinError){
            errorWeightsPopulation.push_back(Mutator::TPGMutator::generateTwinNegErrorWeights(
                errorWeightsPopulation.back()
            ));
            i++;
        }
    }
}


void Learn::EvoStratLearningAgent::doEvolutionStrategy(
    std::multimap<std::shared_ptr<EvaluationResult>, 
                  const std::map<Program::Line*, std::vector<double>>*> results)


{
    std::cout<<std::setprecision(4);
    //std::cout<<std::endl;   
    size_t nbAgentsEval = 500;

    bool firstLine = true;
    size_t iii = 0;

    for (auto &lines : *results.begin()->second) {
        Program::Line *line = lines.first;

        size_t nbUsed = 0;
        
        std::vector<double> originConstants;
        for(auto i=0; i<line->getNbConstants(); i++){
            double* constant = (double*)(line->cGetConstantHandler().getDataAt(typeid(Data::Constant), i).getSharedPointer<Data::Constant>().get());


            originConstants.push_back(*constant);
        }
        std::vector<double> evaluationWeights(line->getNbConstants(), 0);

        // Browse the results
        //std::cout<<"for one line"<<std::endl;
        size_t idxSkip = 0;
        std::vector<double> scores;
        for (const auto &resultEntry : results) {
            if(idxSkip >= nbAgents - nbAgentsEval){
                scores.push_back(resultEntry.first->getResult());
                if(firstLine)std::cout<<"Score"<<resultEntry.first->getResult()<<std::endl;
            }
            idxSkip++;
        }
        
        
        // Get the indices of scores
        std::vector<size_t> indices(scores.size());
        for (size_t i = 0; i < scores.size(); ++i) {
            indices[i] = i;
                if(firstLine)std::cout<<"Index"<<indices[i]<<std::endl;
        }
        // Trier les indices en fonction des valeurs correspondantes dans vect
        std::sort(indices.begin(), indices.end(),
                [&scores](size_t a, size_t b) { return scores[a] < scores[b]; });

        // Créer un vecteur de rangs (même taille que vect)
        std::vector<double> ranks(scores.size());

        // Assigner les rangs en fonction des indices triés
        for (size_t rank = 0; rank < indices.size(); ++rank) {
            ranks[indices[rank]] = static_cast<double>(rank) / (indices.size() - 1);
                if(firstLine)std::cout<<"Ranks"<<ranks[indices[rank]]<<std::endl;
        }


        double meanReward = std::accumulate(ranks.begin(), ranks.end(), 0.0) / ranks.size();
        //std::cout<<"\n\n\n"<<errorsMap.size()<<" "<<meanReward<<std::endl;

        double sq_sum = std::accumulate(ranks.begin(), ranks.end(), 0.0, 
                                        [meanReward](double acc, double val) {
                                            return acc + (val - meanReward) * (val - meanReward);
                                        });
        double stddev = std::sqrt(sq_sum / ranks.size());

        // Assigner les rangs en fonction des indices triés
        /*for (size_t rank = 0; rank < indices.size(); ++rank) {
            ranks[indices[rank]] = (ranks[indices[rank]] - meanReward) / stddev;
                if(firstLine)std::cout<<"Ranks"<<ranks[indices[rank]]<<std::endl;
        }*/

        uint64_t j = 0;
        idxSkip = 0;
        for (const auto &resultEntry : results) {


            if(idxSkip >= nbAgents - nbAgentsEval){
                auto usedLines = resultEntry.first->getUsedLines();
                if(usedLines.find(line) != usedLines.end()){
                    // Get the score result
                    double result = ranks[indices[j]];

                    // Get the error weights
                    const std::vector<double> &errorWeights = resultEntry.second->at(line);
                    if(firstLine)std::cout<<"Result"<<result<<std::endl;

                    // Afficher ou utiliser les valeurs associées à ce programme pour ce résultat
                    uint64_t i = 0;
                    for (double value : errorWeights) {
                        if(firstLine)std::cout<<"Value"<<value<<std::endl;
                        evaluationWeights.at(i) += value * result;
                        if(firstLine)std::cout<<"ErrWe"<<evaluationWeights.at(i)<<std::endl;
                        if(i == -1){
                            if(value < 0){
                            std::cout<<value<<"|";

                            } else {
                                
                                std::cout<<" "<<value<<"|";
                            }

                        }
                        i++;
                    }


                    nbUsed++;
                }


                j++;
            }
            idxSkip++;
        }

        if (nbUsed > 0) {
            for(size_t i = 0; i < line->getNbConstants(); i++){
                double newConstantsValue = originConstants.at(i) + lr * evaluationWeights.at(i) /( (double)(nbUsed) * sigma);
                line->getConstantHandler().setDataAt(typeid(Data::Constant), i, {static_cast<double>(newConstantsValue)});
                    
                    
                    if(firstLine)std::cout<<"OrCon"<<originConstants.at(i)<<std::endl;
                    if(firstLine)std::cout<<"lr"<<lr<<std::endl;
                    if(iii < 20)std::cout<<newConstantsValue<<", ";
                    if(firstLine)std::cout<<"sigma"<<sigma<<std::endl;
                    if(firstLine)std::cout<<"size"<<(double)(nbUsed)<<std::endl;
                    if(firstLine)std::cout<<"NeWCo"<<newConstantsValue<<std::endl;
                    iii++;

            }
        }
        firstLine = false;
    

    }
        //std::cout<<std::endl;
    std::cout<<std::setprecision(2);
}


std::shared_ptr<Learn::EvaluationResult> Learn::EvoStratLearningAgent::evaluateJob(
    TPG::TPGExecutionEngine& tee, const Job& job, uint64_t generationNumber,
    Learn::LearningMode mode, LearningEnvironment& le) const
{

    if(mode == Learn::LearningMode::TRAINING && !falseTraining){
        tee.setErrorWeights(job.getErrorWeights());
        tee.clearUsageLines();
    }

    std::shared_ptr<Learn::EvaluationResult> evaluationResult = LearningAgent::evaluateJob(
        tee, job, generationNumber, mode, le
    );

    if(mode == Learn::LearningMode::TRAINING && !falseTraining){
        evaluationResult->addUsageLines(tee.getUsageLInes());
        evaluationResult->setIndex(job.getIdx());
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
    if (mode == LearningMode::TRAINING && !falseTraining) {
        archiveSeed = this->rng.getUnsignedInt64(0, UINT64_MAX);
    }   
    if (tpgGraph->getNbRootVertices() > 0) {
        if(mode == LearningMode::TRAINING && !falseTraining){
            return std::make_shared<Learn::Job>(
                Learn::Job({vertex}, archiveSeed, idx, &errorWeightsPopulation.at(idx)));
        }
        if(mode == LearningMode::VALIDATION ||falseTraining){
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
