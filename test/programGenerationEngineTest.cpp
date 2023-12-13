/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2021 - 2023) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2021 - 2023)
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

#ifdef CODE_GENERATION
#include <gtest/gtest.h>

#if defined(_MSC_VER) || (__MINGW32__)
// C++17 not available in gcc7 or clang7
#include <filesystem>
#endif

#include "codeGen/programGenerationEngine.h"
#include "data/primitiveTypeArray.h"
#include "environment.h"
#include "goldenReferenceComparison.h"
#include "instructions/addPrimitiveType.h"
#include "instructions/lambdaInstruction.h"
#include "program/program.h"

class ProgramGenerationEngineTest : public ::testing::Test
{
  protected:
    std::vector<std::reference_wrapper<const Data::DataHandler>> vect;
    const size_t size1{32};
    Instructions::Set set;
    Environment* e = nullptr;
    Environment* envWithConstant = nullptr;
    Program::Program* p = nullptr;
    Program::Program* p2 = nullptr;
    Program::Program* p3 = nullptr;

    virtual void SetUp()
    {
        vect.push_back(
            *(new Data::PrimitiveTypeArray<double>((unsigned int)size1)));

        auto add = [](double a, double b) -> double { return a + b; };
        auto addConstant = [](Data::Constant a, double b) -> double {
            return (double)(a) + b;
        };
        auto sub = [](double a, double b) -> double { return a - b; };
        set.add(*(new Instructions::LambdaInstruction<double, double>(
            add, "$0 = $1 + $2;")));
        set.add(*(new Instructions::LambdaInstruction<double, double>(
            sub, "$0 = $1 - $2;")));
        set.add(*(new Instructions::AddPrimitiveType<double>()));

        e = new Environment(set, vect, 8);

        set.add(*(new Instructions::LambdaInstruction<Data::Constant, double>(
            addConstant, "$0 = (double)($1) - $2;")));

        envWithConstant = new Environment(set, vect, 8, 5);
        p = new Program::Program(*e);
        p2 = new Program::Program(*e);
        p3 = new Program::Program(*envWithConstant);

#if defined(_MSC_VER) || (__MINGW32__)
        // Set working directory to BIN_DIR_PATH where the "src" directory was
        // created.
        std::filesystem::current_path(BIN_DIR_PATH);
#endif

        Program::Line& l0 = p->addNewLine();
        l0.setInstructionIndex(0); // Instruction is add.
        // Reg[5] = in1[0] + in1[1];
        l0.setOperand(0, 1, 0);    // 1st operand: parameter 0.
        l0.setOperand(1, 1, 1);    // 2nd operand: parameter 1.
        l0.setDestinationIndex(5); // Destination is register at index 5 (6th)

        Program::Line& l1 = p->addNewLine();
        // Reg[1] = reg[5] + in1[25];
        l1.setInstructionIndex(0); // Instruction is add.
        l1.setOperand(0, 0, 5);    // 1st operand: 6th register.
        l1.setOperand(1, 1, 25);   // 2nd operand: parameter 26.
        l1.setDestinationIndex(1); // Destination is register at index 1

        // Intron line
        Program::Line& l2 = p->addNewLine();
        // Reg[5] = reg[3] - in1[0];
        l2.setInstructionIndex(1); // Instruction is minus.
        l2.setOperand(0, 0, 3);    // 1st operand: 3rd register.
        l2.setOperand(1, 1, 0);    // 2nd operand: parameter 0.
        l2.setDestinationIndex(5); // Destination is register at index 0

        Program::Line& l3 = p->addNewLine();
        // Reg[0] = reg[1] - in1[1];
        l3.setInstructionIndex(1); // Instruction is minus.
        l3.setOperand(0, 0, 1);    // 1st operand: 1st register.
        l3.setOperand(1, 1, 1);    // 2nd operand: 1st parameter.
        l3.setDestinationIndex(0); // Destination is register at index 0

        Program::Line& l4 = p->addNewLine();
        // Reg[0] = reg[0] + in1[5];
        l4.setInstructionIndex(1); // Instruction is minus.
        l4.setOperand(0, 0, 0);    // 1st operand: 0th and 1st registers.
        l4.setOperand(1, 1, 5);    // 2nd operand : parameter 6.
        l4.setDestinationIndex(0); // Destination is register at index 0

        Program::Line& P2l0 = p2->addNewLine();
        P2l0.setInstructionIndex(2); // Instruction is add(not printable).
        // Reg[5] = in1[0] + in1[1];
        P2l0.setOperand(0, 1, 0);    // 1st operand: parameter 0.
        P2l0.setOperand(1, 1, 1);    // 2nd operand: parameter 1.
        P2l0.setDestinationIndex(5); // Destination is register at index 5 (6th)

        Program::Line& P3l0 = p3->addNewLine();
        P3l0.setInstructionIndex(3); // Instruction is add.
        // Reg[5] = cst[0] + in1[1];
        P3l0.setOperand(0, 1, 1);    // 1st operand: constant 0.
        P3l0.setOperand(1, 2, 1);    // 2nd operand: parameter 1.
        P3l0.setDestinationIndex(5); // Destination is register at index 5 (6th)

        // Mark intron lines
        ASSERT_EQ(p->identifyIntrons(), 1);
    }

    virtual void TearDown()
    {
        delete p;
        delete p2;
        delete e;
        delete (&(vect.at(0).get()));
        delete (&set.getInstruction(0));
        delete (&set.getInstruction(1));
        delete (&set.getInstruction(2));
    }
};

TEST_F(ProgramGenerationEngineTest, ConstructorDestructor)
{
    CodeGen::ProgramGenerationEngine* progGen;
    ASSERT_NO_THROW(progGen =
                        new CodeGen::ProgramGenerationEngine("constructor", *e))
        << "Construction failed.";

    ASSERT_NO_THROW(delete progGen) << "Destruction failed.";
    ASSERT_NO_THROW(progGen = new CodeGen::ProgramGenerationEngine(
                        "constructor", *e, "./src/"))
        << "Construction failed.";

    ASSERT_NO_THROW(delete progGen) << "Destruction failed.";

    ASSERT_NO_THROW(progGen =
                        new CodeGen::ProgramGenerationEngine("constructor", *p))
        << "Construction failed with a valid program.";

    ASSERT_NO_THROW(delete progGen) << "Destruction failed.";

    ASSERT_NO_THROW(progGen = new CodeGen::ProgramGenerationEngine(
                        "constructorWithPath", *e, "./src/"))
        << "Failed to construct a TPGGenerationEngine with a filename and a "
           "TPG and a path";

    ASSERT_THROW(progGen = new CodeGen::ProgramGenerationEngine("", *e),
                 std::invalid_argument)
        << "Construction should fail, filename is empty.";

    ASSERT_THROW(progGen = new CodeGen::ProgramGenerationEngine(
                     "constructor", *e, "./src/unkownDir/"),
                 std::runtime_error)
        << "Construction should fail because the path does not exist.";
}

TEST_F(ProgramGenerationEngineTest, generateCurrentLine)
{
    CodeGen::ProgramGenerationEngine* engine =
        new CodeGen::ProgramGenerationEngine("genCurrentLine", *p);

    ASSERT_TRUE(engine != NULL) << "Fail to create a ProgramGenerationEngine.";

    ASSERT_NO_THROW(engine->generateCurrentLine())
        << "Can't generate the first line";

    delete engine; // call the destructor to close the file.

    ASSERT_TRUE(
        compare_files("genCurrentLine.c", TESTS_DAT_PATH
                      "codeGen/ProgramGenerationEngineTest.generateCurrentLine/"
                      "goldenReference.c_ref"))
        << "Error the source file generated is different from the golden "
           "reference.";
    ASSERT_TRUE(
        compare_files("genCurrentLine.h", TESTS_DAT_PATH
                      "codeGen/ProgramGenerationEngineTest.generateCurrentLine/"
                      "goldenReference.h_ref"))
        << "Error the header file generated is different from the golden "
           "reference.";

    engine = new CodeGen::ProgramGenerationEngine("genCurrentLine", *p2);

    ASSERT_TRUE(engine != NULL) << "Fail to create a ProgramGenerationEngine.";

    ASSERT_THROW(engine->generateCurrentLine(), std::runtime_error)
        << "Should not be able to generate line, the instruction is not "
           "printable";

    delete engine;
}

TEST_F(ProgramGenerationEngineTest, generateProgram)
{
    CodeGen::ProgramGenerationEngine engine("genCurrentLine", *p);
    CodeGen::ProgramGenerationEngine engineForConstant("programWithConstant",
                                                       *p3);

    ASSERT_NO_THROW(engine.generateProgram(1))
        << "Out of range exception while generating the program";

    ASSERT_NO_THROW(engine.setProgram(*p2)) << "Fail to set a program";

    ASSERT_THROW(engine.generateProgram(2), std::runtime_error)
        << "Should be able to generate the program contain an instruction not "
           "printable";

    ASSERT_NO_THROW(engineForConstant.generateProgram(3))
        << "Fail to generate a program with constant";
}

TEST_F(ProgramGenerationEngineTest, initOperandCurrentLine)
{

    CodeGen::ProgramGenerationEngine engine("genCurrentLine", *p);

    ASSERT_NO_THROW(engine.generateCurrentLine())
        << "Fail to generate a valid line.";

    ASSERT_NO_THROW(engine.setProgram(*p2)) << "Fail to set a program.";

    ASSERT_THROW(engine.generateCurrentLine(), std::runtime_error)
        << "Should fail to generate a none printable Instruction.";
}
#endif // CODE_GENERATION
