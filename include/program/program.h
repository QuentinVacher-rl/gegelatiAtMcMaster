#ifndef PROGRAM_H
#define PROGRAM_H

#include <vector>

#include "environment.h"
#include "program/line.h"

namespace Program {
	/**
	* \brief The Program class contains a list of program lines that can be executed
	* within a well defined Environment.
	*/
	class Program {
	protected:
		/// Environment within which the Program will be executed.
		const Environment& environment;

		/// Lines of the program stored as raw bits.
		std::vector<Line*> lines;

		/// Delete the default constructor.
		Program() = delete;

	public:
		/**
		* \brief Main constructor of the Program.
		*
		* \param[in] e the reference to the Environment that will be referenced in the Program attributes.
		*/
		Program(const Environment& e) : environment{ e } {
			// Call the computeLineSize() function to check the validity of the environment.
			computeLineSize(this->environment);
		};

		/**
		* \brief Destructor for the Program class.
		*
		* This destructor deallocates all memory allocated for Program lines (if any).
		*/
		~Program();

		/**
		* \brief Static method used to compute the size of Program lines based on information from the Enviroment.
		*
		* The Program line size, expressed in bits, is computed with the following formula:
		* $ceil(log2(n)) + ceil(log2(i)) + m*(ceil(log2(nb_{src}))+ceil(log2(largestAddressSpace)) + p*32$
		* See PROJECT/doc/instructions.md for more details.
		*
		* \param[in] env The Environment whose information is used.
		* \return the computed line size.
		* \throw std::domain_error in cases where the given Environment is
		* parameterized with no registers, contains no Instruction, Instruction
		* with no operands, no DataHandler or DataHandler with no addressable Space.
		*/
		static const size_t computeLineSize(const Environment& env);

		/**
		* \brief Add a new line to the Program with only 0 bits.
		*/
		void addNewLine();

		/**
		* \brief Get the number of lines in the Program.
		*
		* \return the number of line contained in the Program.
		*/
		size_t getNbLines() const;
	};
}
#endif