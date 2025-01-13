/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019 - 2020) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019 - 2020)
 * Nicolas Sourbier <nsourbie@insa-rennes.fr> (2020)
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

#include "mutator/programMutator.h"
#include "mutator/mutationParameters.h"
#include "mutator/rng.h"

void Mutator::ProgramMutator::initRandomProgram(
    Program::Program& p, const MutationParameters& params, Mutator::RNG& rng)
{
    // Empty the program
    while (p.getNbLines() > 0) {
        p.removeLine(0);
    }

    const ProgramParameters& progParams = p.isActionProgram() ? params.actProg : params.contProg;

    // insert random constants in the program
    Data::Constant c_value;
    for (int i = 0; i < p.getEnvironment().getNbConstant(); i++) {
        c_value = {
            rng.getDouble(progParams.minConstValue, progParams.maxConstValue)};
        p.getConstantHandler().setDataAt(typeid(Data::Constant), i, c_value);
    }

    // insert random constants in the program
    double r_value;
    for (int i = 0; i < p.getEnvironment().getNbRegisters() - p.getEnvironment().getParams().nbSharedRegisters; i++) {
        p.getRegisterInitHandler().setDataAt(typeid(double), i, 0.0);
    }

    // Select the number of line randomly
    const uint64_t nbLine = rng.getUnsignedInt64(1, progParams.initProgramSize);
    // Insert them
    while (p.getNbLines() < nbLine) {
        insertRandomLine(p, rng);
    }

    // Identify Introns
    p.identifyIntrons();
}

bool Mutator::ProgramMutator::deleteRandomLine(Program::Program& p,
                                               Mutator::RNG& rng)
{
    // Line cannot be removed from a program with a single line.
    if (p.getNbLines() <= 1) {
        return false;
    }

    uint64_t lineIndex = rng.getUnsignedInt64(0, p.getNbLines() - 1);
    p.removeLine(lineIndex);
    return true;
}

void Mutator::ProgramMutator::insertRandomLine(Program::Program& p,
                                               Mutator::RNG& rng)
{
    uint64_t lineIndex = rng.getUnsignedInt64(0, p.getNbLines());
    Program::Line& line = p.addNewLine(lineIndex);
    Mutator::LineMutator::initRandomCorrectLine(line, rng, p.isActionProgram());
}

bool Mutator::ProgramMutator::swapRandomLines(Program::Program& p,
                                              Mutator::RNG& rng)
{
    if (p.getNbLines() < 2) {
        return false;
    }
    // Select two distinct random index.
    const uint64_t lineIndex0 = rng.getUnsignedInt64(0, p.getNbLines() - 1);
    uint64_t lineIndex1 = rng.getUnsignedInt64(0, p.getNbLines() - 2);
    lineIndex1 += (lineIndex1 >= lineIndex0) ? 1 : 0;

    p.swapLines(lineIndex0, lineIndex1);

    return true;
}

bool Mutator::ProgramMutator::alterRandomLine(Program::Program& p,
                                              Mutator::RNG& rng)
{
    if (p.getNbLines() < 1) {
        return false;
    }
    // Select a random index.
    const uint64_t lineIndex = rng.getUnsignedInt64(0, p.getNbLines() - 1);
    Mutator::LineMutator::alterCorrectLine(p.getLine(lineIndex), rng, p.isActionProgram());
    return true;
}

bool Mutator::ProgramMutator::alterRandomLineConstant(
    Program::Program& p, const MutationParameters& params, Mutator::RNG& rng)
{
    const ProgramParameters& progParams = p.isActionProgram() ? params.actProg : params.contProg;


    const uint64_t line_idx =
        rng.getUnsignedInt64(0, p.getNbLines() - 1);

    if(p.getLine(line_idx).getNbConstants() == 0){
        return false;
    }

    const uint64_t const_idx = rng.getUnsignedInt64(0, p.getLine(line_idx).getNbConstants() - 1);

    Mutator::LineMutator::changeConstantAt(p.getLine(line_idx), const_idx, rng);
    return true;
}


bool Mutator::ProgramMutator::alterRandomConstant(
    Program::Program& p, const MutationParameters& params, Mutator::RNG& rng)
{
    const ProgramParameters& progParams = p.isActionProgram() ? params.actProg : params.contProg;


    const uint64_t register_idx =
        rng.getUnsignedInt64(0, p.getEnvironment().getNbRegisters() - p.getEnvironment().getParams().nbSharedRegisters - 1);
    double currentValue = p.getRegisterInitAt(register_idx);


    if(currentValue == 0.0){
        // Init the value
        currentValue = rng.getDouble(params.tpg.minSharedRegsValue, params.tpg.maxSharedRegsValue);

    } else if (0.5 > rng.getDouble(0.0, 1.0)){

        // Sample the modification factor
        double delta = rng.getDouble(
            0.5, 1.5
        );
        if(delta > 1) delta = delta * 2 - 1;

        // 10% chance of swapping
        if(0.1 > rng.getDouble(0.0, 1.0)) delta *= -1;

        // Compute the new value
        currentValue = currentValue * delta;

    } else {
        currentValue = 0.0;
    }

    // Set it
    p.getRegisterInitHandler().setDataAt(
        typeid(Data::Constant), register_idx,
        {currentValue});

    return true;
}

bool Mutator::ProgramMutator::mutateProgram(Program::Program& p,
                                            const MutationParameters& params,
                                            Mutator::RNG& rng)
{


    const ProgramParameters& progParams = p.isActionProgram() ? params.actProg : params.contProg;



    bool anyMutation = false;
    if (p.getNbLines() > 1 && rng.getDouble(0.0, 1.0) < progParams.pDelete) {
        anyMutation = true;
        deleteRandomLine(p, rng);
    }

    if (p.getNbLines() < progParams.maxProgramSize &&
        rng.getDouble(0.0, 1.0) < progParams.pAdd) {
        anyMutation = true;
        insertRandomLine(p, rng);
    }

    if (rng.getDouble(0.0, 1.0) < progParams.pMutate) {
        anyMutation = true;
        alterRandomLine(p, rng);
    }

    if (rng.getDouble(0.0, 1.0) < progParams.pSwap) {
        anyMutation = true;
        swapRandomLines(p, rng);
    }

    // mutate the programs constants if they exists
    if (//TODO CHANGE THIS IMPORTANT
        rng.getDouble(0.0, 1.0) < progParams.pConstantMutation) {
        anyMutation = true;
        alterRandomLineConstant(p, params, rng);
    }

    // mutate the programs constants if they exists
    if (//TODO CHANGE THIS IMPORTANT
        rng.getDouble(0.0, 1.0) < progParams.pRegsValueMutation) {
        anyMutation = true;
        alterRandomConstant(p, params, rng);
    }

    // Identify introns
    if (anyMutation) {
        p.identifyIntrons();
    }

    return anyMutation;
}
