/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2022) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2022)
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

#ifndef POINTER_WRAPPER_H
#define POINTER_WRAPPER_H

#include "data/constant.h"
#include "data/dataHandler.h"
#include "data/hash.h"
#include "demangle.h"

namespace Data {

    /**
     * DataHandler for manipulating pointers to primitive data type.
     *
     * Contrary to the PrimitiveTypeArray does not contain its data, but
     * possesses a pointer to them.
     *
     */
    template <class T> class PointerWrapper : public DataHandler
    {
        static_assert(std::is_fundamental<T>::value ||
                          std::is_same<T, Data::Constant>(),
                      "Template class PrimitiveTypeArray<T> can only be used "
                      "for primitive types.");

      protected:
        /**
         * \brief Pointer to the data accessed through the PointerWrapper.
         */
        T* containerPtr;

        /**
         * \brief Implementation of the updateHash method.
         *
         * Hash is very lightweight and does not need to be cashed. This method
         * is empty.
         */
        virtual size_t updateHash() const override;

      public:
        /**
         *  \brief Constructor for the PointerWrapper class.
         *
         * \param[in] ptr the pointer managed by the PointerWrapper.
         */
        PointerWrapper(T* ptr = nullptr)
        {
            this->setPointer(ptr);
        };

        /// Default destructor
        virtual ~PointerWrapper() = default;

        /// Default copy constructor.
        PointerWrapper(const PointerWrapper<T>& other) = default;

        // Friend relation needed for copy-construction.
        template <class U> friend class PrimitiveTypeArray;

        /**
         * \brief Return a PrimitiveTypeArray<T> where the data of the
         * PointerWrapper has been copied.
         *
         * \return a PrimitiveTypeArray of size one.
         */
        virtual DataHandler* clone() const override;

        /// Inherited from DataHandler
        virtual bool canHandle(const std::type_info& type) const override;

        /// Inherited from DataHandler
        virtual size_t getAddressSpace(
            const std::type_info& type) const override;

        /// Inherited from DataHandler
        virtual size_t getLargestAddressSpace(void) const override;

        /// Inherited from DataHandler
        virtual size_t getHash() const override;

        /// Inherited from DataHandler. Does nothing.
        void resetData() override;

        /**
         * \brief Set the pointer of the PointerWrapper.
         *
         * This method automatically invalidates the cachedHash.
         *
         * \param[in] ptr the new pointer managed by the PointerWrapper.
         *
         */
        void setPointer(T* ptr);

        /// Inherited from DataHandler
        virtual UntypedSharedPtr getDataAt(const std::type_info& type,
                                           const size_t address) const override;

        /// Inherited from DataHandler
        virtual std::vector<size_t> getAddressesAccessed(
            const std::type_info& type, const size_t address) const override;

#ifdef CODE_GENERATION
        /// Inherited from DataHandler
        virtual const std::type_info& getNativeType() const override;

        /// Inherited from DataHandler
        virtual std::vector<size_t> getDimensionsSize() const override;
#endif
    };

    template <class T> inline size_t PointerWrapper<T>::getHash() const
    {
        return updateHash();
    }

    template <class T> inline size_t PointerWrapper<T>::updateHash() const
    {
        if (this->containerPtr != nullptr) {

            // reset
            this->cachedHash = Data::Hash<size_t>()(this->id);

            // Rotate by 1 because otherwise, xor is comutative.
            this->cachedHash =
                (this->cachedHash >> 1) | (this->cachedHash << 63);

            this->cachedHash ^= Data::Hash<T>()(*this->containerPtr);
            return this->cachedHash;
        }
        else {
            return this->cachedHash = 0;
        }
    }

    // Declare class for clone method
    template <class T> class PrimitiveTypeArray;

    template <class T> inline DataHandler* PointerWrapper<T>::clone() const
    {
        // Create a constantCopy of the PointerWrapper content.
        DataHandler* result = new PrimitiveTypeArray<T>(*this);

        return result;
    }

    template <class T>
    inline bool PointerWrapper<T>::canHandle(const std::type_info& type) const
    {
        if (typeid(T) == type) {
            return true;
        }
        return false;
    }

    template <class T>
    inline size_t PointerWrapper<T>::getAddressSpace(
        const std::type_info& type) const
    {
        if (typeid(T) == type) {
            return 1;
        }
        else {
            return 0;
        }
    }

    template <class T>
    inline size_t PointerWrapper<T>::getLargestAddressSpace(void) const
    {
        return 1;
    }

    template <class T> inline void PointerWrapper<T>::resetData()
    {
        // Does nothing
    }

    template <class T> inline void PointerWrapper<T>::setPointer(T* ptr)
    {
        this->containerPtr = ptr;
        return;
    }

    template <class T>
    inline UntypedSharedPtr PointerWrapper<T>::getDataAt(
        const std::type_info& type, const size_t address) const
    {
        if (this->containerPtr == nullptr) {
            throw std::runtime_error("Null pointer access.");
        }

#ifndef NDEBUG
        // Throw exception in case of invalid arguments.
        if (!this->canHandle(type)) {
            std::stringstream message;
            message << "Data type " << DEMANGLE_TYPEID_NAME(type.name())
                    << " cannot be accessed in a "
                    << DEMANGLE_TYPEID_NAME(typeid(*this).name()) << ".";
            throw std::invalid_argument(message.str());
        }

        if (address > 0) {
            std::stringstream message;
            message << "Data type " << DEMANGLE_TYPEID_NAME(type.name())
                    << " cannot be accessed at address " << address
                    << ", address space size is 1.";
            throw std::out_of_range(message.str());
        }
#endif

        UntypedSharedPtr result(this->containerPtr,
                                UntypedSharedPtr::emptyDestructor<const T>());
        return result;
    }

    template <class T>
    inline std::vector<size_t> PointerWrapper<T>::getAddressesAccessed(
        const std::type_info& type, const size_t address) const
    {
        // Initialize the result
        std::vector<size_t> result;

        // If the accessed address is valid fill the result.
        const size_t space = this->getAddressSpace(type);
        if (space > address) {
            // For the native type.
            if (type == typeid(T)) {
                result.push_back(address);
            }
        }
        return result;
    }

#ifdef CODE_GENERATION
    template <class T>
    inline const std::type_info& PointerWrapper<T>::getNativeType() const
    {
        const std::type_info& a = typeid(T);
        return a;
    }

    template <class T>
    inline std::vector<size_t> PointerWrapper<T>::getDimensionsSize() const
    {
        std::vector<size_t> sizes = {1};
        return sizes;
    }
#endif

} // namespace Data

#endif
