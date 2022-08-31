/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019 - 2022) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019 - 2022)
 * Nicolas Sourbier <nsourbie@insa-rennes.fr> (2019 - 2020)
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

#ifndef TPG_GRAPH_DOT_EXPORTER_H
#define TPG_GRAPH_DOT_EXPORTER_H

#include <map>
#include <stdexcept>
#include <string>

#include "program/line.h"
#include "program/program.h"
#include "tpg/tpgAbstractEngine.h"
#include "tpg/tpgAction.h"
#include "tpg/tpgEdge.h"
#include "tpg/tpgGraph.h"
#include "tpg/tpgTeam.h"
#include "tpg/tpgVertex.h"

namespace File {
    /**
     * \brief Class used to export a TPGGraph into a text file with the dot
     * format.
     */
    class TPGGraphDotExporter : TPG::TPGAbstractEngine
    {
      protected:
        /**
         * \brief File in which the dot content is written during export.
         */
        FILE* pFile;

        /**
         * \brief Character chain used to control the indentation of the
         * exported file.
         */
        std::string offset;

        /**
         * \brief Print the dot content for the given TPGTeam.
         *
         * Content is printed directly into the file opened by the class
         * constructor, or by a call to setNewFilePath.
         *
         * \param[in] team the TPGTeam being printed.
         */
        void printTPGTeam(const TPG::TPGTeam& team);

        /**
         * \brief Print the dot content for the given TPGAction.
         *
         * Content is printed directly into the file opened by the class
         * constructor, or by a call to setNewFilePath.
         * This method returns the identifier associated to the printed action
         * so that the print TPGEdge method can target this TPGAction. Indeed,
         * contrary to TPGTeam which have a unique ID, each action is printed on
         * the fly, with a unique ID, when a TPGEdge is targetting this action.
         *
         * \param[in] action the TPGTeam being printed.
         * \return the identifier associated to this action.
         */
        uint64_t printTPGAction(const TPG::TPGAction& action);

        /**
         * \brief Prints the dot content for the given TPGEdge.
         * Prints the program index followed by its contants if the constants
         * are used.
         *
         * the program is printed under the format :
         * Pindex [fillcolor=\#cccccc shape = point] //constant1|...|constant_n
         *
         * \param[in] edge the TPGEdge being printed.
         */
        void printTPGEdge(const TPG::TPGEdge& edge);

        /**
         * \brief Prints the dot content for the given Program.
         *
         * \param[in] program the Program to be printed
         *
         * a program is stored in the .dot file with the format :
         * line_1&#92;nline2&#92;nline3&#92;n...&#92;nline_N&#92;n
         *
         * a line is stored in the .dot file with the following format
         * inst_idx|dest_idx&op1_param1|op1_param2#...#
         *
         * inst_idx = instruction index
         * dest_idx = destination index
         * op       = operand
         */
        void printProgram(const Program::Program& program);

        /**
         * \brief Prints header content in the dot file.
         *
         * This method prints preliminary content that must be printed into the
         * dot file before any vertex or edge.
         */
        void printTPGGraphHeader();

        /**
         * \brief Prints footer content in the dot file.
         *
         * This method prints finalization content that must be printed into the
         * dot file after all vertices and edges.
         */
        void printTPGGraphFooter();

      public:
        /**
         * \brief Constructor for the exporter.
         *
         * \param[in] filePath initial path to the file where the dot content
         * will be written.
         * \param[in] graph const reference to the graph whose content will
         * be exported in dot.
         * \throws std::runtime_error in case no file could be opened at the
         * given filePath.
         */
        TPGGraphDotExporter(const char* filePath, const TPG::TPGGraph& graph)
            : TPG::TPGAbstractEngine(graph), pFile{NULL}, offset{""}
        {
            if ((pFile = fopen(filePath, "w")) == NULL) {
                throw std::runtime_error("Could not open file " +
                                         std::string(filePath));
            }
        };

        /**
         * Disable copy construction.
         *
         * Until we see the need for it, there si no reason to enable
         * copy-construction of TPGGraphDotExporter.
         */
        TPGGraphDotExporter(const TPGGraphDotExporter& other) = delete;

        /**
         * Disable TPGGraphDotExporter default assignment operator.
         *
         * Until we see the need for it, there si no reason to enable assignment
         * operator of TPGGraphDotExporter.
         */
        TPGGraphDotExporter& operator=(const TPGGraphDotExporter& other) =
            delete;

        /**
         * Destructor for the exporter.
         *
         * Closes the file.
         */
        ~TPGGraphDotExporter()
        {
            if (pFile != NULL) {
                fclose(pFile);
            }
        }

        /**
         * \brief Set a new file for the exporter.
         *
         * \param[in] newFilePath new path to the file where the dot content
         * will be written.
         * \throws std::runtime_error in case no file could be opened at the
         * given newFilePath.
         */
        void setNewFilePath(const char* newFilePath)
        {
            //  Close previous file
            fclose(pFile);

            // open new one;
            if ((pFile = fopen(newFilePath, "w")) == NULL) {
                pFile = NULL;
                throw std::runtime_error("Could not open file " +
                                         std::string(newFilePath));
            }
        }

        /**
         * \brief Print the TPGGraph given when constructing the
         * TPGGraphDotExporter into a dot file.
         */
        void print();

        /**
         * \brief Print a sub-tree of the TPGGraph given when constructing the
         * TPGGraphDotExporter into a dot file.
         *
         * Contrary to the print() method, which prints the whole TPG, this
         * method only prints the TPG stemming from the TPG::TPGVertex passed as
         * a parameter. Hence, only vertices and programs connected to this
         * TPGVertex will be printed in the file, and all others will be
         * ignored.
         *
         * \param[in] root The vertex used as a starting point to print a
         * connected TPG.
         */
        void printSubGraph(const TPG::TPGVertex* root);
    };
}; // namespace File

#endif
