#include <gtest/gtest.h>
#include "instructionAdd.h"

TEST(InstructionAdd, ConstructorDestructorCall) {
	Instruction* i = new InstructionAdd<double>();
	ASSERT_NE(i, nullptr) << "Call to constructor for InstructionAdd<double> failed.";
	delete i;
	i = new InstructionAdd<int>();
	ASSERT_NE(i, nullptr) << "Call to constructor for InstructionAdd<int> failed.";
	delete i;
}

TEST(InstructionAdd, OperandListAndNbParam) {
	Instruction* i = new InstructionAdd<double>();
	auto operands = i->getOperandTypes();
	ASSERT_EQ(operands.size(), 2) << "Operand list of InstructionAdd<double> is different from 2";
	ASSERT_STREQ(operands.at(0).get().name(), typeid(PrimitiveType<double>).name()) << "First operand of InstructionAdd<double> is not\"" << typeid(PrimitiveType<double>).name() << "\".";
	ASSERT_STREQ(operands.at(1).get().name(), typeid(PrimitiveType<double>).name()) << "Second operand of InstructionAdd<double> is not\"" << typeid(PrimitiveType<double>).name() << "\".";
	ASSERT_EQ(i->getNbParameters(), 0) << "Number of parameters of InstructionAdd<double> should be 0.";
	delete i;
}

TEST(InstructionAdd, CheckArgumentTypes) {
	Instruction* i = new InstructionAdd<double>();
	PrimitiveType<double> a{ 2.5 };
	PrimitiveType<double> b = 5.6;
	PrimitiveType<double> c = 3.7;
	PrimitiveType<int> d = 5;

	std::vector<std::reference_wrapper<SupportedType>> vect;
	vect.push_back(a);
	vect.push_back(b);
	ASSERT_TRUE(i->checkOperandTypes(vect)) << "Operands of valid types wrongfully classified as invalid.";
	vect.push_back(c);
	ASSERT_FALSE(i->checkOperandTypes(vect)) << "Operands list of too long size wrongfully classified as valid.";
	vect.pop_back();
	vect.pop_back();
	vect.push_back(d);
	ASSERT_FALSE(i->checkOperandTypes(vect)) << "Operands of invalid types wrongfully classified as valid";
	delete i;
}