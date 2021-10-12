/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2021) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2021)
 * Thomas Bourgoin <tbourgoi@insa-rennes.fr> (2021)
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

/// doc in ../README.md
#include <stdio.h>

#include "StickGameBest_TPG.h"

#define ERROR_INFERENCE 1
#define ERROR_RESET 2
int* in1;
int* in2;
int main(int argc, char* argv[])
{
    int remainingSticks[1];
    int hints[4];
    in1 = hints;
    in2 = remainingSticks;
    int expectedVal;
    int action;
    if (argc == 1) {
        expectedVal = -1;
        remainingSticks[0] = 21;
        hints[0] = 1;
        hints[1] = 2;
        hints[2] = 3;
        hints[3] = 4;
    }
    else {
        //        expectedVal = atoi(argv[1]);
        int cpt = 0, i = 1;
        for (; i < argc && cpt < 3; ++cpt, ++i) {
            hints[cpt] = atoi(argv[i]);
        }
        remainingSticks[0] = atoi(argv[i]);
    }

    action = executeFromVertex(root);
    return action;
#ifdef DEBUG
    printf("action : %d\n", action);
#endif // DEBUG
    if (expectedVal != -1 && action != expectedVal) {
        return ERROR_INFERENCE;
    }

    reset();
    action = executeFromVertex(root);
    if (expectedVal != -1 && action != expectedVal) {
        return ERROR_RESET;
    }
    return 0;
}