/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019 - 2021) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019 - 2020)
 * Nicolas Sourbier <nsourbie@insa-rennes.fr> (2020)
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

#include <gtest/gtest.h>
#include <vector>

#include "data/dataHandler.h"
#include "data/primitiveTypeArray.h"
#include "data/primitiveTypeArray2D.h"
#include "data/untypedSharedPtr.h"
#include "instructions/addPrimitiveType.h"
#include "instructions/lambdaInstruction.h"
#include "instructions/multByConstant.h"
#include "instructions/set.h"
#include "program/line.h"
#include "program/program.h"
#include "program/programExecutionEngine.h"

class ProgramExecutionEngineTest : public ::testing::Test
{
  protected:
    const size_t size1{24};
    const size_t size2{32};
    const double value0{2.3};
    const float value1{1.2f};
    const double value2{0.5};
    const double value3{1.5};
    std::vector<std::reference_wrapper<const Data::DataHandler>> vect;
    Instructions::Set set;
    Environment* e;
    Program::Program* p;

    virtual void SetUp()
    {
        vect.push_back(
            *(new Data::PrimitiveTypeArray<int>((unsigned int)size1)));
        vect.push_back(
            *(new Data::PrimitiveTypeArray<double>((unsigned int)size2)));
        vect.push_back(*(new Data::PrimitiveTypeArray2D<double>(size1, size2)));

        ((Data::PrimitiveTypeArray<double>&)vect.at(1).get())
            .setDataAt(typeid(double), 25, value0);
        ((Data::PrimitiveTypeArray<double>&)vect.at(1).get())
            .setDataAt(typeid(double), 5, value2);
        ((Data::PrimitiveTypeArray<double>&)vect.at(1).get())
            .setDataAt(typeid(double), 6, value3);
        ((Data::PrimitiveTypeArray2D<double>&)vect.at(2).get())
            .setDataAt(typeid(double), 0, value0);
        ((Data::PrimitiveTypeArray2D<double>&)vect.at(2).get())
            .setDataAt(typeid(double), 1, value1);
        ((Data::PrimitiveTypeArray2D<double>&)vect.at(2).get())
            .setDataAt(typeid(double), 24, value0);
        ((Data::PrimitiveTypeArray2D<double>&)vect.at(2).get())
            .setDataAt(typeid(double), 25, value0);

        set.add(*(new Instructions::AddPrimitiveType<double>()));
        set.add(*(new Instructions::MultByConstant<double>()));
        set.add(*new Instructions::LambdaInstruction<const double[2],
                                                     const double[2]>(
            [](const double a[2], const double b[2]) {
                return a[0] * b[0] + a[1] * b[1];
            }));
        set.add(*new Instructions::LambdaInstruction<const double[2][2]>(
            [](const double a[2][2]) {
                double res = 0.0;
                for (auto h = 0; h < 2; h++) {
                    for (auto w = 0; w < 2; w++) {
                        res += a[h][w];
                    }
                }
                return res / 4.0;
            }));

        e = new Environment(set, vect, 8, 5);
        p = new Program::Program(*e);

        Program::Line& l0 = p->addNewLine();
        l0.setInstructionIndex(
            3); // Instruction is lambdaInstruction<double[2][2]>.
        l0.setOperand(0, 4, 0);    // 1st operand: 4 values in 2D array
        l0.setDestinationIndex(5); // Destination is register at index 5 (6th)

        Program::Line& l1 = p->addNewLine();
        l1.setInstructionIndex(0); // Instruction is addPrimitiveType<double>.
        l1.setOperand(0, 0, 5);    // 1st operand: 6th register.
        l1.setOperand(1, 3, 25);   // 2nd operand: 26th double in the
                                   // PrimitiveTypeArray of double.
        l1.setDestinationIndex(1); // Destination is register at index 1

        // Intron line
        Program::Line& l2 = p->addNewLine();
        l2.setInstructionIndex(1); // Instruction is MultByConstant<double>.
        l2.setOperand(0, 0, 3);    // 1st operand: 3rd register.
        l2.setOperand(1, 1, 0);    // 2nd operand: parameter 0.
        p->getConstantHandler().setDataAt(
            typeid(Data::Constant), 0,
            {static_cast<double>(
                value0)});         // Parameter is set to value1 (=2.3f) => 2
        l2.setDestinationIndex(0); // Destination is register at index 0

        Program::Line& l3 = p->addNewLine();
        l3.setInstructionIndex(1); // Instruction is MultByConstant<double>.
        l3.setOperand(0, 0, 1);    // 1st operand: 1st register.
        l3.setOperand(1, 1, 1);    // 2nd operand: 1st parameter.
        p->getConstantHandler().setDataAt(
            typeid(Data::Constant), 1,
            {static_cast<double>(
                value1)});         // Parameter is set to value1 (=1.2f) => 1
        l3.setDestinationIndex(0); // Destination is register at index 0

        Program::Line& l4 = p->addNewLine();
        l4.setInstructionIndex(
            2);                 // Instruction is LambdaInstruction<double[2]>.
        l4.setOperand(0, 0, 0); // 1st operand: 0th and 1st registers.
        l4.setOperand(1, 3, 5); // 2nd operand : 6th and 7th double in the
                                // PrimitiveTypeArray of double.
        l4.setDestinationIndex(0); // Destination is register at index 0

        // Mark intron lines
        ASSERT_EQ(p->identifyIntrons(), 1);
    }

    virtual void TearDown()
    {
        delete p;
        delete e;
        delete (&(vect.at(0).get()));
        delete (&(vect.at(1).get()));
        delete (&(vect.at(2).get()));
        delete (&set.getInstruction(0));
        delete (&set.getInstruction(1));
        delete (&set.getInstruction(2));
        delete (&set.getInstruction(3));
    }
};

TEST_F(ProgramExecutionEngineTest, ConstructorDestructor)
{
    Program::ProgramExecutionEngine* progExecEng;
    ASSERT_NO_THROW(progExecEng = new Program::ProgramExecutionEngine(*p))
        << "Construction failed.";

    ASSERT_NO_THROW(delete progExecEng) << "Destruction failed.";

    std::vector<std::reference_wrapper<Data::DataHandler>> vect2;
    vect2.push_back(*vect.at(0).get().clone());
    ASSERT_THROW(progExecEng = new Program::ProgramExecutionEngine(*p, vect2),
                 std::runtime_error)
        << "Construction should faile with data sources differing in number "
           "from those of the Environment.";
    vect2.push_back(*vect.at(1).get().clone());

    ASSERT_NO_THROW(progExecEng = new Program::ProgramExecutionEngine(*p))
        << "Construction failed with a perfect copy of the environment data "
           "source.";
    ASSERT_NO_THROW(delete progExecEng) << "Destruction failed.";

    // Push a new dataHandler instead.
    // Because its id is different, it will not be accepted by the PEE.
    delete (&(vect2.at(1).get()));
    vect2.pop_back();
    vect2.push_back(
        *(new Data::PrimitiveTypeArray<double>((unsigned int)size2)));
    ASSERT_THROW(progExecEng = new Program::ProgramExecutionEngine(*p, vect2),
                 std::runtime_error)
        << "Construction should fail with data sources differing in id from "
           "those of the Environment.";

    delete (&(vect2.at(0).get()));
    delete (&(vect2.at(1).get()));
}

TEST_F(ProgramExecutionEngineTest, executeCurrentLine)
{
    Program::ProgramExecutionEngine progExecEng(*p);

    ASSERT_NO_THROW(progExecEng.executeCurrentLine())
        << "Execution of the first line of the program from Fixture should not "
           "fail.";
    progExecEng.next();
    ASSERT_NO_THROW(progExecEng.executeCurrentLine())
        << "Execution of the second line of the program from Fixture should "
           "not "
           "fail.";
    progExecEng.next(); // Skips the intron automatically
    ASSERT_NO_THROW(progExecEng.executeCurrentLine())
        << "Execution of the third line of the program from Fixture should "
           "not fail.";
    progExecEng.next();
    ASSERT_NO_THROW(progExecEng.executeCurrentLine())
        << "Execution of the fourth line of the program from Fixture should "
           "not "
           "fail.";
    progExecEng.next();
    ASSERT_THROW(progExecEng.executeCurrentLine(), std::out_of_range)
        << "Execution of a non-existing line of the program should fail.";
}

TEST_F(ProgramExecutionEngineTest, execute)
{
    Program::ProgramExecutionEngine progExecEng(*p);
    double result;

    double r6 = (value0 + value1 + value0 + value0) / 4;
    double r1 = value0 + r6;
    double r0 = r1 * (value1);
    r0 = r0 * value2 + r1 * value3;

    ASSERT_NO_THROW(result = progExecEng.executeProgram())
        << "Program from fixture failed to execute. (Indivitual execution of "
           "its line in executeCurrentLine test).";
    ASSERT_EQ(result, r0)
        << "Result of the program from Fixture is not as expected.";

    // Introduce a new line in the program to test the throw
    Program::Line& l5 = p->addNewLine();
    // Instruction 4 does not exist. Must deactivate checks to write this
    // instruction
    l5.setInstructionIndex(4, false);
    ASSERT_THROW(progExecEng.executeProgram(), std::out_of_range)
        << "Program line using a incorrect Instruction index should throw an "
           "exception.";

    // Now ignoring the exceptions
    ASSERT_NO_THROW(result = progExecEng.executeProgram(true))
        << "Program line using a incorrect Instruction index should not "
           "interrupt the Execution when ignored.";
    ASSERT_EQ(result, r0) << "Result of the program from Fixture, with an "
                             "additional ignored line, is not as expected.";
}
