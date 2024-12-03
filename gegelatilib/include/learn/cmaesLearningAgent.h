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

#ifndef CMAES_LEARNING_AGENT_H
#define CMAES_LEARNING_AGENT_H

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <functional>

#include "learn/parallelEvoStratLearningAgent.h"
#include "log/esBasicLogger.h"

using namespace Eigen;

namespace Learn {

    /**
     * \brief Class used to control the learning steps of a TPGGraph within
     * a given LearningEnvironment.
     */
    class CMAESLearningAgent : public ParallelEvoStratLearningAgent
    {

        protected:

            int N, lambda,  counteval, eigeneval;
            double sigma, chiN, mu, cm;
            double mueff=0.0;
            double cs=0.0;
            double cc=0.0;
            double c1=0.0;
            double cmu=0.0;
            double damps=0.0;
            VectorXd xmean, ps, pc, weights;
            MatrixXd B, D, C, arx, arz;
            std::vector<double> arfitness;
            std::vector<int> arindex;

            std::set<Program::Line*> lineUsed;

            
            void initializeWeights();

            // Generate lambda offspring
            void generation();

            // Evaluate offspring fitness
            void evaluation();

            // Update internal parameters
            void update();

            // CMA-ES parameters
            void compute_coefs();
            
            uint64_t computeDimension(LearningAgent& la);
            void initMeanValues();
            void updateLineUsed();

        public:

            /**
             * \brief Constructor for CMAESLearningAgent.
             *
             * CMAESLearningAgent must be created with an existing learningAgent.
             *
             * \param[in] la The LearningAgent used.
             */
            CMAESLearningAgent(ParallelLearningAgent& la)
                : ParallelEvoStratLearningAgent(la),
                N(computeDimension(la)), sigma(0.1), cm(1),
                lambda(4 + floor(3 * log(N))),
                mu(lambda / 2),
                counteval(0), eigeneval(0),
                xmean(VectorXd::Zero(N)),
                ps(VectorXd::Zero(N)), pc(VectorXd::Zero(N)),
                B(MatrixXd::Identity(N, N)), D(MatrixXd::Identity(N, N)),
                C(B * D * (B*D).transpose()),
                chiN(std::sqrt((double)N) * (1.0 - 1.0 / (4.0 * (double)N) + 1.0 / (21.0 * (double)N * (double)N)))
                { 
                    updateLineUsed();
                    initializeWeights(); 
                    compute_coefs();
                    initMeanValues();
                }

            /**
             * \brief TODO
             */
            virtual void generateErrorWeights() override;

            /**
             * \brief Do the evolution strategy depending on the results 
             * 
             * \param[in] results TODO
            */ 
            virtual void doEvolutionStrategy(std::multimap<std::shared_ptr<EvaluationResult>, const std::map<Program::Line*, std::vector<double>>*> results) override;

            virtual void updateRoot();


    };
}; // namespace Learn

#endif
