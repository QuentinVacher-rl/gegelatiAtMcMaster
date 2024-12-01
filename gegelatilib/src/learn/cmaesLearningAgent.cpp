

#include "learn/cmaesLearningAgent.h"
using namespace Eigen;


void Learn::CMAESLearningAgent::initializeWeights() {
    weights = VectorXd::Zero(mu);
    for (int i = 0; i < mu; ++i) {
        weights(i) = log(mu + 0.5) - log(i + 1);
    }
    weights /= weights.sum();
}

// Generate lambda offspring
void Learn::CMAESLearningAgent::generation() {

    

    std::mt19937 generator(rng.getInt32(0, 10000000)); // 42 est la graine
    std::normal_distribution<double> distribution(0, 1);

    arz.resize(N, lambda);
    arx.resize(N, lambda);
    for (int k = 0; k < lambda; ++k) {
        for (int i = 0; i < N; ++i) {
            arz(i, k) = distribution(generator);
        }
        arx.col(k) = xmean + sigma * (B * D * arz.col(k));
    }
}

// Evaluate offspring fitness
void Learn::CMAESLearningAgent::evaluation() {
    arfitness.resize(lambda);
    for (int k = 0; k < lambda; ++k) {
        //arfitness[k] = felli(arx.col(k));
        ++counteval;
    }
    arindex.resize(lambda);
    std::iota(arindex.begin(), arindex.end(), 0);
    std::sort(arindex.begin(), arindex.end(), [&](int i, int j) {
        return arfitness[i] < arfitness[j];
    });
}

// Update internal parameters
void Learn::CMAESLearningAgent::update() {
    VectorXd xold = xmean;
    xmean = VectorXd::Zero(N);
    VectorXd zmean = VectorXd::Zero(N);
    for (int i = 0; i < mu; ++i) {
        xmean += weights(i) * arx.col(arindex[i]);
        zmean += weights(i) * arz.col(arindex[i]);
    }

    // Update evolution paths
    ps = (1 - cs()) * ps + sqrt(cs() * (2 - cs()) * mueff()) * B * zmean;
    bool hsig = ps.norm() / sqrt(1 - pow(1 - cs(), 2 * counteval / lambda)) / chiN < 1.4 + 2 / (N + 1);
    pc = (1 - cc()) * pc + hsig * sqrt(cc() * (2 - cc()) * mueff()) * B * D * zmean;

    // Adapt covariance matrix
    C = (1 - c1() - cmu()) * C
        + c1() * (pc * pc.transpose() + (1 - hsig) * cc() * (2 - cc()) * C)
        + cmu() * (B * D * arz.leftCols(mu)) * weights.asDiagonal() * (B * D * arz.leftCols(mu)).transpose();

    // Adapt step size
    sigma *= exp((cs() / damps()) * (ps.norm() / chiN - 1));

    // Update B and D from C
    if (counteval % (lambda / 10) == 0) {
        SelfAdjointEigenSolver<MatrixXd> eigensolver(C);
        B = eigensolver.eigenvectors();
        D = eigensolver.eigenvalues().cwiseSqrt().asDiagonal();
    }
}

// CMA-ES parameters
double Learn::CMAESLearningAgent::mueff() const {
    return weights.sum() * weights.sum() / weights.squaredNorm();
}


double Learn::CMAESLearningAgent::cs() const {
    return (mueff() + 2) / (N + mueff() + 5);
}


double Learn::CMAESLearningAgent::cc() const {
    return (4 + mueff() / N) / (N + 4 + 2 * mueff() / N);
}


double Learn::CMAESLearningAgent::c1() const {
    return 2 / (std::pow(N + 1.3, 2) + mueff());
}


double Learn::CMAESLearningAgent::cmu() const {
    return std::min(1 - c1(), 2 * (mueff() - 2 + 1 / mueff()) / (std::pow(N + 2, 2) + 2 * mueff() / 2));
}


double Learn::CMAESLearningAgent::damps() const {
    return 1 + 2 * std::max(0.0, std::sqrt((mueff() - 1) / (N + 1)) - 1) + cs();
}


uint64_t Learn::CMAESLearningAgent::computeDimension(Learn::LearningAgent& la)
{    
    uint64_t nbDimensions = 0;
    lineUsed.clear();

    std::vector<const std::list<std::unique_ptr<TPG::TPGEdge>>*> allEdges;
    allEdges.push_back(&la.getTPGGraph()->getEdges());
    allEdges.push_back(&la.getTPGGraph()->getActionEdges());

    for (const auto* edgeList : allEdges) {
        // Assurez-vous que edgeList est un pointeur vers une liste de unique_ptr
        for (const auto& edge : *edgeList) {
            
            Program::Program* program = &edge->getProgram();

            for(size_t idx_line = 0; idx_line< program->getNbLines(); idx_line++){

                Program::Line* line = &program->getLine(idx_line);

                if(!program->isIntron(idx_line)){

                    nbDimensions += line->getNbConstants();
                    lineUsed.push_back(line);
                }
            }


        }  
    }

    return nbDimensions;
}


VectorXd Learn::CMAESLearningAgent::initMeanValues(LearningAgent& la)
{
    std::vector<double> listValues;

    std::vector<const std::list<std::unique_ptr<TPG::TPGEdge>>*> allEdges;
    allEdges.push_back(&la.getTPGGraph()->getEdges());
    allEdges.push_back(&la.getTPGGraph()->getActionEdges());

    for (const auto* edgeList : allEdges) {
        // Assurez-vous que edgeList est un pointeur vers une liste de unique_ptr
        for (const auto& edge : *edgeList) {
            
            Program::Program* program = &edge->getProgram();

            for(size_t idx_line = 0; idx_line< program->getNbLines(); idx_line++){

                Program::Line* line = &program->getLine(idx_line);

                if(!program->isIntron(idx_line)){

                    for(size_t idx_const = 0; idx_const < line->getNbConstants(); idx_const++){
                        listValues.push_back(line->getConstantAt(idx_const));
                    }
                }
            }
        }  
    }

    VectorXd initialMean = Map<VectorXd>(listValues.data(), listValues.size());
    return initialMean;
}


void Learn::CMAESLearningAgent::generateErrorWeights(){

    errorWeightsPopulation.clear();
    this->generation();

    for(size_t idxAgent=0; idxAgent < lambda; idxAgent++){
        
        std::map<Program::Line*, std::vector<double>> errorWeights;

        size_t idxConstant = 0;
        for(Program::Line* line: lineUsed){

            std::vector<double> errorThisLine(line->getNbConstants());
            for(size_t idxLine=0; idxLine < line->getNbConstants(); idxLine++){
                errorThisLine.push_back(arx(idxAgent, idxConstant));
                idxConstant++;
            }

            errorWeights.insert(std::make_pair(line, errorThisLine));
        }  

        errorWeightsPopulation.push_back(errorWeights);
    }
}

/**
 * \brief Do the evolution strategy depending on the results 
 * 
 * \param[in] results TODO
*/ 

void Learn::CMAESLearningAgent::doEvolutionStrategy(std::multimap<std::shared_ptr<Learn::EvaluationResult>, const std::map<Program::Line*, std::vector<double>>*> results){
    
    errorWeightsPopulation;

    arfitness.resize(lambda);
    int k = 0;
    for (auto pair: errorWeightsPopulation) {
        // Find the index of the pair in results
        auto it = std::find_if(results.begin(), results.end(),
            [&pair](const auto& resultPair) {
                return resultPair.second == &pair; // Compare pointers
            });

        if (it != results.end()) {

            arfitness[k] = it->first->getResult(); 
        } else {
            // If the pair is not found
            std::cout << "Pair not found in results." << std::endl;
        }

        ++counteval;
    }


    arindex.resize(lambda);
    std::iota(arindex.begin(), arindex.end(), 0);
    std::sort(arindex.begin(), arindex.end(), [&](int i, int j) {
        return arfitness[i] < arfitness[j];
    });
    this->update();
    this->updateRoot();
}

void Learn::CMAESLearningAgent::updateRoot()
{    
    std::vector<const std::list<std::unique_ptr<TPG::TPGEdge>>*> allEdges;
    allEdges.push_back(&tpg->getEdges());
    allEdges.push_back(&tpg->getActionEdges());


    size_t idx_meanVal = 0;
    for (const auto* edgeList : allEdges) {
        // Assurez-vous que edgeList est un pointeur vers une liste de unique_ptr
        for (const auto& edge : *edgeList) {
            
            Program::Program* program = &edge->getProgram();

            for(size_t idx_line = 0; idx_line< program->getNbLines(); idx_line++){

                Program::Line* line = &program->getLine(idx_line);

                if(!program->isIntron(idx_line)){

                    for(size_t idx_const = 0; idx_const < line->getNbConstants(); idx_const++){
                        line->getConstantHandler().setDataAt(
                            typeid(Data::Constant), idx_const, {static_cast<double>(xmean[idx_meanVal])
                        });
                        idx_meanVal++;
                    }
                }
            }
        }  
    }

}

