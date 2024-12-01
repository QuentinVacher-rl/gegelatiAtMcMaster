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

            int N, lambda, mu, counteval;
            double sigma, chiN;
            VectorXd xmean, ps, pc, weights;
            MatrixXd B, D, C, arx, arz;
            std::vector<double> arfitness;
            std::vector<int> arindex;

            std::vector<Program::Line*> lineUsed;

            
            void initializeWeights();

            // Generate lambda offspring
            void generation();

            // Evaluate offspring fitness
            void evaluation();

            // Update internal parameters
            void update();

            // CMA-ES parameters
            double mueff() const;

            double cs() const;

            double cc() const;

            double c1() const;

            double cmu() const;

            double damps() const;

            uint64_t computeDimension(LearningAgent& la);

            VectorXd initMeanValues(LearningAgent& la);

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
                N(computeDimension(la)), sigma(sigma),
                xmean(initMeanValues(la)),
                lambda(4 + floor(3 * log(N))),
                mu(lambda / 2),
                counteval(0),
                ps(VectorXd::Zero(N)), pc(VectorXd::Zero(N)),
                B(MatrixXd::Identity(N, N)), D(MatrixXd::Identity(N, N)),
                C(B * D * B.transpose()),
                chiN(std::sqrt(N) * (1 - 1 / (4.0 * N) + 1 / (21.0 * N * N)))
                { initializeWeights(); }

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
