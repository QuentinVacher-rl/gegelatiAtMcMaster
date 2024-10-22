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

#ifndef EVO_STRAT_LEARNING_AGENT_H
#define EVO_STRAT_LEARNING_AGENT_H

#include "learn/learningAgent.h"

namespace Learn {

    /**
     * \brief Class used to control the learning steps of a TPGGraph within
     * a given LearningEnvironment.
     */
    class EvoStratLearningAgent : public LearningAgent
    {
      protected:

        /**
         * vector containing, for each value a map.
         * 
         * Each map link each program in the graph to a vector of error weights applied to the constant of the program.
         */
        std::vector<std::map<std::shared_ptr<Program::Program>, std::vector<double>>> errorWeightsPopulation;


      public:

        /**
         * \brief Constructor for EvoStratLearningAgent.
         *
         * EvoStratLearningAgent must be created with an existing learningAgent.
         *
         * \param[in] la The LearningAgent used.
         */
        EvoStratLearningAgent(LearningAgent& la)
            : LearningAgent(la){};

        /**
         * \brief Train the TPGGraph for one generation.
         *
         * Training for one generation includes:
         * - Populating the TPGGraph according to given MutationParameters.
         * - Evaluating all roots of the TPGGraph. (call to evaluateAllRoots)
         * - Removing from the TPGGraph the worst performing root TPGVertex.
         *
         * \param[in] generationNumber the integer number of the current
         * generation.
         */
        virtual void trainOneGeneration(uint64_t generationNumber);

        /**
         * \brief Takes a given TPGVertex and creates a job containing it.
         * Useful for example in adversarial mode where a job could contain a
         * match of several roots.
         *
         * \param[in] vertex the TPGVertex stemming a TPGGraph to be evaluated.
         * \param[in] mode the mode of the training, determining for example
         * if we generate values that we only need for training.
         * \param[in] idx The index of the job, can be used to organize a map
         * for example.
         * \param[in] tpgGraph The TPG graph from which we will take the
         * root.
         *
         * \return A job representing the root.
         */
        virtual std::shared_ptr<Learn::Job> makeJob(
            const TPG::TPGVertex* vertex, Learn::LearningMode mode, int idx = 0,
            TPG::TPGGraph* tpgGraph = nullptr);

    };
}; // namespace Learn

#endif
