

#include "learn/cmaesLearningAgent.h"
using namespace Eigen;


void Learn::CMAESLearningAgent::initializeWeights() {
    weights = VectorXd::Zero(mu);
    for (int i = 0; i < mu; ++i) {
        weights(i) = log(mu + 0.5) - log(i + 1);
    }
    weights /= weights.sum();
    mu = floor(mu);

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

        /*if(twinError && k < lambda - 1){
            arz.col(k+1) = -arz.col(k);
            arx.col(k+1) = xmean + sigma * (B * D * arz.col(k+1));
            k++;
        }*/
    }
}

void Learn::CMAESLearningAgent::update() {
    //std::cout<<std::setprecision(4);
    // Affichage des hyperparamètres
    /*std::cout << "Hyperparameters at the start of update():\n";
    std::cout << "mueff: " << mueff << std::endl;
    std::cout << "cs: " << cs << std::endl;
    std::cout << "cc: " << cc << std::endl;
    std::cout << "c1: " << c1 << std::endl;
    std::cout << "cmu: " << cmu << std::endl;
    std::cout << "damps: " << damps << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "sigma: " << sigma << std::endl;
    std::cout << "xmean: " << xmean.transpose() << std::endl;
    std::cout << "lambda: " << lambda << std::endl;
    std::cout << "mu: " << mu << std::endl;*/

    VectorXd xold = xmean;
    VectorXd zmean = VectorXd::Zero(N);
    //xmean = VectorXd::Zero(N);
    for (int i = 0; i < mu; ++i) {
        //xmean += cm/sigma * weights(i) * (arz.col(arindex[i]) - xold);
        zmean += weights(i) * arz.col(arindex[i]);
    }
    xmean = xmean + sigma * (B * D * zmean);




    // Update evolution paths
    ps = (1 - cs) * ps + (sqrt(cs * (2.0 - cs) * mueff)) * (B * zmean);
    bool hsig = ps.norm() / sqrt(1.0 - pow(1.0 - cs, 2.0 * (double)counteval / (double)lambda)) / chiN < 1.4 + 2.0 / ((double)N + 1.0);
    pc = (1.0 - cc) * pc + hsig * sqrt(cc * (2.0 - cc) * mueff) * (B * D * zmean);

    // Affichage des chemins d'évolution
    //std::cout << "ps norm: " << ps.norm() << ", ps: " << ps.transpose() << std::endl;
    //std::cout << "pc norm: " << pc.norm() << ", pc: " << pc.transpose() << std::endl;

    MatrixXd sorted_arz = MatrixXd(N, (int)mu);
    for (int i = 0; i < mu; ++i) {
        sorted_arz.col(i) = arz.col(arindex[i]);
    }

    // Mise à jour de C avec toutes les contributions
    C = (1.0 - c1 - cmu) * C
        + c1 * (pc * pc.transpose() + (1.0 - hsig) * cc * (2.0 - cc) * C)
        + cmu * (B * D * sorted_arz) * weights.asDiagonal() * (B * D * sorted_arz).transpose();

    // Affichage de la matrice de covariance
    //std::cout << "Covariance matrix C: \n" << C << std::endl;

    // Add regularization to ensure numerical stability
    const double epsilon = 1e-8;
    C += epsilon * MatrixXd::Identity(N, N);

    std::cout<<"\n"<<C<<std::endl;

    // Adapt step size
    sigma *= exp((cs / damps) * (ps.norm() / chiN - 1));
    //sigma = std::max(sigma, 0.2);

    // Affichage de la taille de pas (step size)
    // std::cout << "Step size sigma: " << sigma << std::endl;

    // Update B and D from C conditionally
    if (true || counteval - eigeneval > (double)lambda / ((c1 + cmu) / (double)N / 10.0)) {
        eigeneval = counteval;
        
        // Symmetry enforcement again for numerical stability
        C = 0.5 * (C + C.transpose());

        // Eigen decomposition
        SelfAdjointEigenSolver<MatrixXd> eigensolver(C);
        B = eigensolver.eigenvectors();

        if ((eigensolver.eigenvalues().array() < 0).any()) {
            std::cout << "La matrice n'est pas positive semi-définie !" << std::endl;
        }

        // Ensure non-negative eigenvalues
        VectorXd eigenvalues = eigensolver.eigenvalues();
        eigenvalues = eigenvalues.cwiseMax(0);
        //std::cout << "Eigenvalues of C: " << eigenvalues.transpose() << std::endl;



        // Compute D from eigenvalues
        D = eigenvalues.cwiseSqrt().asDiagonal();

        // Affichage des valeurs propres et des matrices B et D
        //std::cout << "Eigenvalues of C: " << eigenvalues.transpose() << std::endl;
        //std::cout << "Matrix B (eigenvectors): \n" << B << std::endl;
        //std::cout << "Matrix D (eigenvalues sqrt): \n" << D << std::endl;
        /*std::cout << "Eigenvalues of C: " << eigensolver.eigenvalues().transpose() << std::endl;
        std::cout << "ps norm: " << ps.norm() << std::endl;
        std::cout << "pc norm: " << pc.norm() << std::endl;
        std::cout << "sigma: " << sigma << std::endl;
        std::cout << "Fitness best: " << arfitness[arindex[0]] << std::endl;*/
    }


    // Affichage du compteur d'évaluations
    /*std::cout << "Counteval: " << counteval << std::endl;
    std::cout << "Step size sigma: " << sigma << std::endl;
    std::cout<<std::setprecision(2);*/
}

// CMA-ES parameters
void Learn::CMAESLearningAgent::compute_coefs() {
    mueff = weights.sum() * weights.sum() / weights.squaredNorm();
    cc = (4.0 + mueff / (double)N) / ((double)N + 4.0 + 2.0 * mueff / (double)N);
    cs = (mueff + 2.0) / ((double)N + mueff + 5.0);
    c1 = 2.0 / (std::pow((double)N + 1.3, 2.0) + mueff);
    cmu = std::min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / (std::pow((double)N + 2.0, 2.0) + 2.0 * mueff / 2.0));
    damps = 1.0 + 2.0 * std::max(0.0, std::sqrt((mueff - 1.0) / ((double)N + 1.0)) - 1.0) + cs;

}


void Learn::CMAESLearningAgent::doEvolutionStrategy(std::multimap<std::shared_ptr<Learn::EvaluationResult>, const std::map<Program::Line*, std::vector<double>>*> results){

    //std::cout<<std::endl;
    arfitness.resize(lambda);
    for (auto pair: results) {
        // Minimize the score
        arfitness[pair.first->getIndex()] = -pair.first->getResult(); 
        ++counteval;
    }

    arindex.resize(lambda);
    std::iota(arindex.begin(), arindex.end(), 0);
    std::sort(arindex.begin(), arindex.end(), [&](int i, int j) {
        return arfitness[i] < arfitness[j];
    });

    /*std::cout<<"\nIndex: ";
    for(auto idx: arindex){
        std::cout<<idx<<" ";
    }std::cout<<std::endl;
    std::cout<<"\nFitness: ";
    for(auto idx: arfitness){
        std::cout<<idx<<" ";
    }std::cout<<std::endl;*/

    this->update();
    this->updateRoot();
}


void Learn::CMAESLearningAgent::generateErrorWeights(){

    errorWeightsPopulation.clear();
    this->generation();

    for(size_t idxAgent=0; idxAgent < lambda; idxAgent++){
        
        std::map<Program::Line*, std::vector<double>> errorWeights;

        size_t idxConstant = 0;
        for(Program::Line* line: lineUsed){

            std::vector<double> errorThisLine;
            for(size_t idxLine=0; idxLine < line->getNbConstants(); idxLine++){
                errorThisLine.push_back(arx(idxConstant, idxAgent));
                idxConstant++;
            }

            errorWeights.insert(std::make_pair(line, errorThisLine));
        }

        errorWeightsPopulation.push_back(errorWeights);
    }
}



uint64_t Learn::CMAESLearningAgent::computeDimension(Learn::LearningAgent& la)
{    

    uint64_t nbDimensions = 0;

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
                }
            }


        }  
    }


    return nbDimensions;
}

void Learn::CMAESLearningAgent::updateLineUsed()
{    

    lineUsed.clear();

    std::vector<const std::list<std::unique_ptr<TPG::TPGEdge>>*> allEdges;
    allEdges.push_back(&tpg->getEdges());
    allEdges.push_back(&tpg->getActionEdges());

    for (const auto* edgeList : allEdges) {
        // Assurez-vous que edgeList est un pointeur vers une liste de unique_ptr
        for (const auto& edge : *edgeList) {
            
            Program::Program* program = &edge->getProgram();

            for(size_t idx_line = 0; idx_line< program->getNbLines(); idx_line++){

                Program::Line* line = &program->getLine(idx_line);

                if(!program->isIntron(idx_line)){

                    lineUsed.insert(line);
                }
            }


        }  
    }
}




void Learn::CMAESLearningAgent::initMeanValues()
{
    std::vector<double> listValues;

    for(auto line: this->lineUsed){
        for(size_t idx_const = 0; idx_const < line->getNbConstants(); idx_const++){
            listValues.push_back(line->getConstantAt(idx_const));
        }
    }

    xmean = Map<VectorXd>(listValues.data(), listValues.size());
}

void Learn::CMAESLearningAgent::updateRoot()
{    
    std::vector<const std::list<std::unique_ptr<TPG::TPGEdge>>*> allEdges;
    allEdges.push_back(&tpg->getEdges());
    allEdges.push_back(&tpg->getActionEdges());


    size_t idx_meanVal = 0;
    for(auto line: this->lineUsed){
        for(size_t idx_const = 0; idx_const < line->getNbConstants(); idx_const++){
            double currentWeight = (double)line->getConstantAt(idx_const);
            line->getConstantHandler().setDataAt(
                typeid(Data::Constant), idx_const, {static_cast<double>(
                    (1-cm) * currentWeight + cm * xmean[idx_meanVal])
            });
            idx_meanVal++;
        }
    }

}

