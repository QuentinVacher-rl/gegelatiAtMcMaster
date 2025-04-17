

#include "tpg/tpgAgent.h"

Program::Program& TPG::TPGAgent::getProgram(const TPG::TPGEdge& edge) const
{
    return *this->programs.at(&edge);
}

void TPG::TPGAgent::setProgram(
    const TPG::TPGEdge* edge,
    std::shared_ptr<Program::Program> prog)
    
{
    this->programs[edge] = prog;
}

std::shared_ptr<Program::Program> TPG::TPGAgent::getProgramSharedPointer(TPG::TPGEdge* edge) const
{
    return this->programs.at(edge);
}


bool TPG::TPGAgent::deletePair(TPG::TPGEdge* edge)
{
    auto it = this->programs.find(edge);
    
    if(it != this->programs.end()){
        this->programs.erase(edge);
        return true;
    }
    return false;
}

const std::unordered_map<const TPG::TPGEdge*, std::shared_ptr<Program::Program>>& TPG::TPGAgent::getPrograms()
{
    return this->programs;
}

const TPG::TPGVertex* TPG::TPGAgent::getRootSpecies() const
{
    return this->rootSpecies;
}

size_t TPG::TPGAgent::agentSize() const
{
    return this->programs.size();
}

void TPG::TPGAgent::setToBeDeleted(bool status){
    this->toBeDeleted = status;
}

bool TPG::TPGAgent::isToBeDeleted() const{
    return this->toBeDeleted;
}