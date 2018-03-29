/************************************************************************/
/*                                                                      */
/*     Copyright 2017-2018 by Ullrich Koethe                            */
/*                                                                      */
/*    This file is part of the XVIGRA image analysis library.           */
/*                                                                      */
/*    Permission is hereby granted, free of charge, to any person       */
/*    obtaining a copy of this software and associated documentation    */
/*    files (the "Software"), to deal in the Software without           */
/*    restriction, including without limitation the rights to use,      */
/*    copy, modify, merge, publish, distribute, sublicense, and/or      */
/*    sell copies of the Software, and to permit persons to whom the    */
/*    Software is furnished to do so, subject to the following          */
/*    conditions:                                                       */
/*                                                                      */
/*    The above copyright notice and this permission notice shall be    */
/*    included in all copies or substantial portions of the             */
/*    Software.                                                         */
/*                                                                      */
/*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
/*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
/*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
/*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
/*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
/*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
/*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
/*    OTHER DEALINGS IN THE SOFTWARE.                                   */
/*                                                                      */
/************************************************************************/

#ifndef XVIGRA_TINY_VECTOR_HPP
#define XVIGRA_TINY_VECTOR_HPP

#include <algorithm>
#include <xtensor/xbuffer_adaptor.hpp>

#include "global.hpp"
#include "error.hpp"
#include "concepts.hpp"
#include "math.hpp"

namespace xvigra
{
    /***********************/
    /* forward declaration */
    /***********************/

    template <class VALUETYPE, index_t N, class REPRESENTATION>
    class tiny_vector_impl;

    /***************/
    /* tiny_vector */
    /***************/

    /* Adds common functionality to the respective tiny_vector_impl */
    template <class VALUETYPE, index_t N, class REPRESENTATION>
    class tiny_vector
    : public tiny_vector_impl<VALUETYPE, N, REPRESENTATION>
    {
      public:

        using self_type = tiny_vector<VALUETYPE, N, REPRESENTATION>;
        using base_type = tiny_vector_impl<VALUETYPE, N, REPRESENTATION>;
        using value_type = typename base_type::value_type;
        using const_value_type = typename base_type::const_value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using iterator = typename base_type::iterator;
        using const_iterator = typename base_type::const_iterator;
        using reverse_iterator = typename base_type::reverse_iterator;
        using const_reverse_iterator = typename base_type::const_reverse_iterator;
        using size_type = typename base_type::size_type;
        using difference_type = typename base_type::difference_type;

        using base_type::owns_memory;
        using base_type::has_fixed_size;
        using base_type::static_size;

        using base_type::base_type;

        tiny_vector();
        tiny_vector(tiny_vector const & rhs);
        tiny_vector(tiny_vector && rhs);

        template <class T, index_t M, class R>
        tiny_vector(tiny_vector<T, M, R> const & rhs);

        template <class T, class A>
        tiny_vector(std::vector<T, A> const & v);

        template <class T, std::size_t M>
        tiny_vector(std::array<T, M> const & v);

        template <class T, std::size_t M, class A, bool I>
        tiny_vector(xt::svector<T, M, A, I> const & v);

        template <class T, class A>
        tiny_vector(xt::uvector<T, A> const & v);

        tiny_vector & operator=(tiny_vector const & rhs);
        tiny_vector & operator=(tiny_vector && rhs);

        tiny_vector & operator=(value_type const & v);

        template <class T, class A>
        tiny_vector & operator=(std::vector<T, A> const & v);

        template <class T, std::size_t M>
        tiny_vector & operator=(std::array<T, M> const & v);

        template <class T, std::size_t M, class A, bool I>
        tiny_vector & operator=(xt::svector<T, M, A, I> const & v);

        template <class T, class A>
        tiny_vector & operator=(xt::uvector<T, A> const & v);

        template <class U, index_t M, class R>
        tiny_vector & operator=(tiny_vector<U, M, R> const & rhs);

        using base_type::resize;
        using base_type::assign;

        void assign(std::initializer_list<value_type> v);

        using base_type::data;

        using base_type::operator[];
        reference at(size_type i);
        constexpr const_reference at(size_type i) const;

        reference front();
        reference back();
        constexpr const_reference front() const;
        constexpr const_reference back()  const;

        template <index_t FROM, index_t TO>
        auto subarray();
        template <index_t FROM, index_t TO>
        auto subarray() const;
        auto subarray(size_type FROM, size_type TO);
        auto subarray(size_type FROM, size_type TO) const;

        auto erase(size_type m) const;
        auto pop_front() const;
        auto pop_back() const;

        auto insert(size_type m, value_type v) const;
        auto push_front(value_type v) const;
        auto push_back(value_type v) const;

        using base_type::begin;
        constexpr const_iterator cbegin() const;
        iterator end();
        constexpr const_iterator end() const;
        constexpr const_iterator cend() const;

        using base_type::rbegin;
        constexpr const_reverse_iterator crbegin() const;
        reverse_iterator rend();
        constexpr const_reverse_iterator rend() const;
        constexpr const_reverse_iterator crend() const;

        using base_type::size;
        using base_type::max_size;
        constexpr bool empty() const;

        using base_type::swap;

            /// factory functions for the k-th unit vector
        template <index_t SIZE=static_size>
        static auto unit_vector(index_t k);

        static auto unit_vector(index_t size, index_t k);

            /// factory function for fixed-size linear sequence ending at <tt>end-1</tt>
        static auto range(value_type end);

            /// factory function for a linear sequence from <tt>begin</tt> to <tt>end</tt>
            /// (exclusive) with stepsize <tt>step</tt>
        template <class T1, class T2=value_type>
        static auto range(value_type begin, T1 end, T2 step = value_type(1));
    };

    /************************************/
    /* default dynamic tiny_vector_impl */
    /************************************/

    template <class VALUETYPE>
    class tiny_vector_impl<VALUETYPE, runtime_size, void>
    : public tiny_vector_impl<VALUETYPE, runtime_size, VALUETYPE[4]>
    {
        using base_type = tiny_vector_impl<VALUETYPE, runtime_size, VALUETYPE[4]>;
      public:
        using base_type::base_type;
    };

    /************************************************/
    /* tiny_vector_impl: dynamic shape, owns memory */
    /************************************************/

    template <class VALUETYPE, index_t BUFFER_SIZE>
    class tiny_vector_impl<VALUETYPE, runtime_size, VALUETYPE[BUFFER_SIZE]>
    : public tags::tiny_vector_tag
    {
      public:
        using self_type = tiny_vector_impl<VALUETYPE, runtime_size, VALUETYPE[BUFFER_SIZE]>;
        using representation_type = VALUETYPE *;
        using buffer_type = VALUETYPE[BUFFER_SIZE < 1 ? 1 : BUFFER_SIZE];
        using allocator_type = std::allocator<VALUETYPE>;

        using value_type = VALUETYPE;
        using const_value_type = typename std::add_const<value_type>::type;
        using reference = value_type &;
        using const_reference = const_value_type &;
        using pointer = value_type *;
        using const_pointer = const_value_type *;
        using iterator = value_type *;
        using const_iterator = const_value_type *;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        constexpr static bool owns_memory = true;
        constexpr static bool has_fixed_size = false;
        constexpr static index_t static_size = runtime_size;
        constexpr static index_t buffer_size = BUFFER_SIZE;

        template <class NEW_VALUETYPE>
        using rebind = tiny_vector<NEW_VALUETYPE, runtime_size, NEW_VALUETYPE[BUFFER_SIZE]>;

        template <index_t NEW_SIZE>
        using rebind_size = tiny_vector<value_type, NEW_SIZE < runtime_size ? runtime_size : NEW_SIZE>;

        tiny_vector_impl();
        ~tiny_vector_impl();

        explicit tiny_vector_impl(size_type n);
        tiny_vector_impl(size_type n, const value_type& v);
        tiny_vector_impl(size_type n, tags::skip_initialization_tag);

        template <class IT,
                  VIGRA_REQUIRE<input_iterator_concept<IT>::value>>
        tiny_vector_impl(IT begin, IT end);

        tiny_vector_impl(std::initializer_list<value_type> const & v);

        tiny_vector_impl(tiny_vector_impl const & v);
        tiny_vector_impl(tiny_vector_impl && v);

        tiny_vector_impl & operator=(tiny_vector_impl const & v);
        tiny_vector_impl & operator=(tiny_vector_impl && v);

        void assign(size_type n, const value_type& v);

        template <class IT,
                  VIGRA_REQUIRE<input_iterator_concept<IT>::value>>
        void assign(IT other_begin, IT other_end);

        reference operator[](size_type i);
        constexpr const_reference operator[](size_type i) const;

        pointer data();
        constexpr const_pointer data() const;

        void resize(size_type n);

        size_type capacity() const;
        size_type size() const;
        size_type max_size() const;
        bool on_stack() const;

        iterator begin();
        const_iterator begin() const;

        reverse_iterator rbegin();
        const_reverse_iterator rbegin() const;

        void swap(tiny_vector_impl & other);

        constexpr static bool may_use_uninitialized_memory = xt::xtrivially_default_constructible<value_type>::value;

      protected:
        /* allocate() assumes that m_size is already set,
           but no memory has been allocated yet */
        void allocate(value_type const & v = value_type());
        void allocate(tags::skip_initialization_tag);
        template <class IT,
                  VIGRA_REQUIRE<input_iterator_concept<IT>::value>>
        void allocate(IT other_begin);

        void deallocate();

        allocator_type m_allocator;
        size_type m_size;
        representation_type m_data;
        buffer_type m_buffer;
    };

    /****************************************/
    /* default fixed shape tiny_vector_impl */
    /****************************************/

    template <class VALUETYPE, index_t N>
    class tiny_vector_impl<VALUETYPE, N, void>
    : public tiny_vector_impl<VALUETYPE, N, std::array<VALUETYPE, (std::size_t)N>>
    {
        using base_type = tiny_vector_impl<VALUETYPE, N, std::array<VALUETYPE, (std::size_t)N>>;
      public:
        using base_type::base_type;
    };

    /**********************************************/
    /* tiny_vector_impl: fixed shape, owns memory */
    /**********************************************/

    template <class VALUETYPE, index_t N>
    class tiny_vector_impl<VALUETYPE, N, std::array<VALUETYPE, (size_t)N>>
    : public std::array<VALUETYPE, (size_t)N>
    , public tags::tiny_vector_tag
    {
      public:
        using base_type = std::array<VALUETYPE, (size_t)N>;
        using self_type = tiny_vector_impl<VALUETYPE, N, base_type>;

        using value_type = VALUETYPE;
        using const_value_type = typename std::add_const<value_type>::type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using iterator = typename base_type::iterator;
        using const_iterator = typename base_type::const_iterator;
        using reverse_iterator = typename base_type::reverse_iterator;
        using const_reverse_iterator = typename base_type::const_reverse_iterator;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        constexpr static bool owns_memory = true;
        constexpr static bool has_fixed_size = true;
        constexpr static index_t static_size = N;

        template <class NEW_VALUETYPE, index_t NEW_SIZE=N>
        using rebind = tiny_vector<NEW_VALUETYPE, NEW_SIZE < runtime_size ? runtime_size : NEW_SIZE>;

        template <index_t NEW_SIZE>
        using rebind_size = tiny_vector<value_type, NEW_SIZE < runtime_size ? runtime_size : NEW_SIZE>;

        tiny_vector_impl();

        explicit tiny_vector_impl(size_type n);
        explicit tiny_vector_impl(tags::skip_initialization_tag);
        tiny_vector_impl(size_type n, const value_type& v);
        tiny_vector_impl(size_type n, tags::skip_initialization_tag);

        template <class IT,
                  VIGRA_REQUIRE<input_iterator_concept<IT>::value>>
        explicit tiny_vector_impl(IT begin);

        template <class IT,
                  VIGRA_REQUIRE<input_iterator_concept<IT>::value>>
        tiny_vector_impl(IT begin, IT end);

        tiny_vector_impl(std::initializer_list<value_type> const & v);

        tiny_vector_impl(tiny_vector_impl const & v);
        tiny_vector_impl(tiny_vector_impl && v);

        tiny_vector_impl & operator=(tiny_vector_impl const & v);
        tiny_vector_impl & operator=(tiny_vector_impl && v);

        void resize(std::size_t s);
        void assign(size_type n, const value_type& v);

        template <class IT,
                  VIGRA_REQUIRE<input_iterator_concept<IT>::value>>
        void assign(IT other_begin, IT other_end);

        using base_type::operator[];
        using base_type::data;

        using base_type::size;
        using base_type::max_size;
        constexpr size_type capacity() const;

        using base_type::begin;
        using base_type::cbegin;
        using base_type::rbegin;
        using base_type::crbegin;
        using base_type::end;
        using base_type::cend;
        using base_type::rend;
        using base_type::crend;

        using base_type::swap;
    };

    /******************************/
    /* representation type traits */
    /******************************/

    namespace tiny_vector_detail
    {

        template <class T>
        struct test_value_type
        {
            static void test(...);

            template <class U>
            static typename U::value_type test(U *, typename U::value_type * = 0);

            constexpr static bool value = !std::is_same<decltype(test((T*)0)), void>::value;
        };

        template <class T,
                  bool has_embedded_types=test_value_type<T>::value,
                  bool is_iterator=random_access_iterator_concept<T>::value>
        struct representation_type_traits;

        template <class T>
        struct representation_type_traits<T, true, false> // T is a container
        {
            using value_type = typename T::value_type;
            using iterator = typename T::iterator;
            using const_iterator = typename T::const_iterator;
            using reverse_iterator = typename T::reverse_iterator;
            using const_reverse_iterator = typename T::const_reverse_iterator;
        };

        template <class T>
        struct representation_type_traits<T, true, true> // T is an iterator
        {
            using value_type = typename T::value_type;
            using iterator = T;
            using const_iterator = T;
            using reverse_iterator = std::reverse_iterator<T>;
            using const_reverse_iterator = std::reverse_iterator<T>;
        };

        template <class T>
        struct representation_type_traits<T *, false, true>
        {
            using value_type             = T;
            using iterator               = value_type *;
            using const_iterator         = value_type const *;
            using reverse_iterator       = std::reverse_iterator<iterator>;
            using const_reverse_iterator = std::reverse_iterator<const_iterator>;
        };

        template <class T>
        struct representation_type_traits<T const *, false, true>
        {
            using value_type             = T const;
            using iterator               = value_type *;
            using const_iterator         = value_type *;
            using reverse_iterator       = std::reverse_iterator<iterator>;
            using const_reverse_iterator = std::reverse_iterator<const_iterator>;
        };
    }

    /**************************************************/
    /* tiny_vector_impl: fixed shape, borrowed memory */
    /**************************************************/

    template <class VALUETYPE, index_t N, class REPRESENTATION>
    class tiny_vector_impl
    : public tags::tiny_vector_tag
    {
        using traits = tiny_vector_detail::representation_type_traits<REPRESENTATION>;
        using deduced_value_type = std::remove_const_t<typename traits::value_type>;
        static_assert(std::is_same<std::remove_const_t<VALUETYPE>, deduced_value_type>::value,
                      "tiny_vector_impl: type mismatch between VALUETYPE and REPRESENTATION.");

      public:
        using representation_type = REPRESENTATION;
        using self_type = tiny_vector_impl<VALUETYPE, N, representation_type>;

        using value_type = VALUETYPE;
        using const_value_type = typename std::add_const<value_type>::type;
        using reference              = value_type &;
        using const_reference        = const_value_type &;
        using pointer                = value_type *;
        using const_pointer          = const_value_type *;
        using iterator               = typename traits::iterator;
        using const_iterator         = typename traits::const_iterator;
        using reverse_iterator       = typename traits::reverse_iterator;
        using const_reverse_iterator = typename traits::const_reverse_iterator;
        using size_type              = std::size_t;
        using difference_type        = std::ptrdiff_t;

        constexpr static bool owns_memory = false;
        constexpr static bool has_fixed_size = true;
        constexpr static index_t static_size = N;

        tiny_vector_impl();

        explicit tiny_vector_impl(representation_type const & begin);
        tiny_vector_impl(representation_type const & begin, representation_type const & end);

        template <class IT,
                  VIGRA_REQUIRE<input_iterator_concept<IT>::value>>
        tiny_vector_impl(IT begin, IT end);

        tiny_vector_impl(tiny_vector_impl const & v) = default;
        tiny_vector_impl(tiny_vector_impl && v) = default;

        tiny_vector_impl & operator=(tiny_vector_impl const & v) = default;
        tiny_vector_impl & operator=(tiny_vector_impl && v) = default;

        void reset(representation_type const & begin);

        void resize(std::size_t s);
        void assign(size_type n, const value_type& v);

        template <class IT,
                  VIGRA_REQUIRE<input_iterator_concept<IT>::value>>
        void assign(IT other_begin, IT other_end);

        reference operator[](size_type i);
        constexpr const_reference operator[](size_type i) const;

        pointer data();
        constexpr const_pointer data() const;

        constexpr size_type size() const;
        constexpr size_type max_size() const;
        constexpr size_type capacity() const;

        iterator begin();
        constexpr const_iterator begin() const;

        reverse_iterator rbegin();
        constexpr const_reverse_iterator rbegin() const;

        void swap(tiny_vector_impl &);

      protected:

        representation_type m_data;
    };

    /****************************************************/
    /* tiny_vector_impl: dynamic shape, borrowed memory */
    /****************************************************/

    template <class VALUETYPE, class REPRESENTATION>
    class tiny_vector_impl<VALUETYPE, runtime_size, REPRESENTATION>
    : public tags::tiny_vector_tag
    {
        using traits = tiny_vector_detail::representation_type_traits<REPRESENTATION>;
        using deduced_value_type = std::remove_const_t<typename traits::value_type>;
        static_assert(std::is_same<std::remove_const_t<VALUETYPE>, deduced_value_type>::value,
                      "tiny_vector_impl: type mismatch between VALUETYPE and REPRESENTATION.");

      public:
        using representation_type = REPRESENTATION;
        using self_type = tiny_vector_impl<VALUETYPE, runtime_size, representation_type>;

        using value_type = VALUETYPE;
        using const_value_type = typename std::add_const<value_type>::type;
        using reference              = value_type &;
        using const_reference        = const_value_type &;
        using pointer                = value_type *;
        using const_pointer          = const_value_type *;
        using iterator               = typename traits::iterator;
        using const_iterator         = typename traits::const_iterator;
        using reverse_iterator       = typename traits::reverse_iterator;
        using const_reverse_iterator = typename traits::const_reverse_iterator;
        using size_type              = std::size_t;
        using difference_type        = std::ptrdiff_t;

        constexpr static bool owns_memory = false;
        constexpr static bool has_fixed_size = false;
        constexpr static index_t static_size = runtime_size;

        tiny_vector_impl();

        tiny_vector_impl(representation_type const & begin, representation_type const & end);

        template <class IT,
                  VIGRA_REQUIRE<input_iterator_concept<IT>::value>>
        tiny_vector_impl(IT begin, IT end);

        tiny_vector_impl(tiny_vector_impl const & v) = default;
        tiny_vector_impl(tiny_vector_impl && v) = default;

        tiny_vector_impl & operator=(tiny_vector_impl const & v) = default;
        tiny_vector_impl & operator=(tiny_vector_impl && v) = default;

        void reset(representation_type const & begin, representation_type const & end);

        void resize(std::size_t s);
        void assign(size_type n, const value_type& v);

        template <class IT,
                  VIGRA_REQUIRE<input_iterator_concept<IT>::value>>
        void assign(IT other_begin, IT other_end);

        reference operator[](size_type i);
        constexpr const_reference operator[](size_type i) const;

        pointer data();
        constexpr const_pointer data() const;

        constexpr size_type size() const;
        constexpr size_type max_size() const;
        constexpr size_type capacity() const;

        iterator begin();
        constexpr const_iterator begin() const;

        reverse_iterator rbegin();
        constexpr const_reverse_iterator rbegin() const;

        void swap(tiny_vector_impl &);

      protected:

        size_type m_size;
        representation_type m_data;
    };

    /********************************************************/
    /* tiny_vector_impl: dynamic shape, xt::xbuffer_adaptor */
    /********************************************************/

    template <class VALUETYPE, class CP, class O, class A>
    class tiny_vector_impl<VALUETYPE, runtime_size, xt::xbuffer_adaptor<CP, O, A>>
    : public xt::xbuffer_adaptor<CP, O, A>
    , public tags::tiny_vector_tag
    {
        using deduced_value_type = typename xt::xbuffer_adaptor<CP, O, A>::value_type;
        static_assert(std::is_same<VALUETYPE, deduced_value_type>::value,
                      "tiny_vector_base: type mismatch between VALUETYPE and REPRESENTATION.");
      public:
        using base_type              = xt::xbuffer_adaptor<CP, O, A>;
        using self_type              = tiny_vector_impl<VALUETYPE, runtime_size, base_type>;
        using value_type             = VALUETYPE;
        using const_value_type       = typename std::add_const<value_type>::type;
        using reference              = value_type &;
        using const_reference        = const_value_type &;
        using pointer                = value_type *;
        using const_pointer          = const_value_type *;
        using iterator               = typename base_type::iterator;
        using const_iterator         = typename base_type::const_iterator;
        using reverse_iterator       = typename base_type::reverse_iterator;
        using const_reverse_iterator = typename base_type::const_reverse_iterator;
        using size_type              = std::size_t;
        using difference_type        = std::ptrdiff_t;

        constexpr static bool owns_memory = false;
        constexpr static bool has_fixed_size = false;
        constexpr static index_t static_size = runtime_size;

        using base_type::base_type;

        void resize(std::size_t s);
        void assign(size_type n, const value_type& v);

        template <class IT,
                  VIGRA_REQUIRE<input_iterator_concept<IT>::value>>
        void assign(IT other_begin, IT other_end);

        using base_type::operator[];
        using base_type::data;
        using base_type::size;

        constexpr size_type max_size() const;
        constexpr size_type capacity() const;

        using base_type::begin;
        using base_type::cbegin;
        using base_type::rbegin;
        using base_type::crbegin;
        using base_type::end;
        using base_type::cend;
        using base_type::rend;
        using base_type::crend;

        using base_type::swap;
        void swap(tiny_vector_impl &);
    };

    template <class V, index_t N, class R>
    inline void
    swap(tiny_vector<V, N, R> & l, tiny_vector<V, N, R> & r)
    {
        l.swap(r);
    }

    /**********************/
    /* tiny_vector output */
    /**********************/

    template <class T, index_t N, class R>
    std::ostream & operator<<(std::ostream & o, tiny_vector<T, N, R> const & v)
    {
        o << "{";
        if(v.size() > 0)
            o << promote_type_t<T>(v[0]);
        for(decltype(v.size()) i=1; i < v.size(); ++i)
            o << ", " << promote_type_t<T>(v[i]);
        o << "}";
        return o;
    }

    /**************************/
    /* tiny_vector comparison */
    /**************************/

    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline bool
    operator==(tiny_vector<V1, N1, R1> const & l,
               tiny_vector<V2, N2, R2> const & r)
    {
        return l.size() == r.size() && std::equal(l.cbegin(), l.cend(), r.cbegin());
    }

    template <class V1, index_t N1, class R1, class V2,
              VIGRA_REQUIRE<!tiny_vector_concept<V2>::value &&
                              std::is_convertible<V2, V1>::value> >
    inline bool
    operator==(tiny_vector<V1, N1, R1> const & l,
               V2 const & r)
    {
        auto i = l.cbegin();
        for(decltype(l.size()) k=0; k < l.size(); ++k, ++i)
        {
            if(*i != r)
            {
                return false;
            }
        }
        return true;
    }

    template <class V1, class V2, index_t N2, class R2,
              VIGRA_REQUIRE<!tiny_vector_concept<V1>::value &&
                              std::is_convertible<V2, V1>::value> >
    inline bool
    operator==(V1 const & l,
               tiny_vector<V2, N2, R2> const & r)
    {
        return r == l;
    }

    template <class V1, index_t N1, class R1, class V2>
    inline bool
    operator!=(tiny_vector<V1, N1, R1> const & l,
               V2 const & r)
    {
        return !(l == r);
    }

    template <class V1, class V2, index_t N2, class R2,
              VIGRA_REQUIRE<!tiny_vector_concept<V1>::value &&
                              std::is_convertible<V2, V1>::value> >
    inline bool
    operator!=(V1 const & l,
               tiny_vector<V2, N2, R2> const & r)
    {
        return !(r == l);
    }

        /// lexicographical less
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline bool
    operator<(tiny_vector<V1, N1, R1> const & l,
              tiny_vector<V2, N2, R2> const & r)
    {
        return std::lexicographical_compare(l.begin(), l.end(), r.begin(), r.end());
    }

        /// lexicographical less-equal
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline bool
    operator<=(tiny_vector<V1, N1, R1> const & l,
               tiny_vector<V2, N2, R2> const & r)
    {
        return !(r < l);
    }

        /// lexicographical greater
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline bool
    operator>(tiny_vector<V1, N1, R1> const & l,
              tiny_vector<V2, N2, R2> const & r)
    {
        return r < l;
    }

        /// lexicographical greater-equal
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline bool
    operator>=(tiny_vector<V1, N1, R1> const & l,
               tiny_vector<V2, N2, R2> const & r)
    {
        return !(l < r);
    }

        /// check if all elements are non-zero (or 'true' if V is bool)
    template <class V, index_t N, class R>
    inline bool
    all(tiny_vector<V, N, R> const & t)
    {
        for(decltype(t.size()) i=0; i<t.size(); ++i)
            if(t[i] == V())
                return false;
        return true;
    }

        /// check if at least one element is non-zero (or 'true' if V is bool)
    template <class V, index_t N, class R>
    inline bool
    any(tiny_vector<V, N, R> const & t)
    {
        for(decltype(t.size()) i=0; i<t.size(); ++i)
            if(t[i] != V())
                return true;
        return false;
    }

    /**********************************/
    /* pointwise relational operators */
    /**********************************/

    #define XVIGRA_TINY_COMPARISON(NAME, OP)                                   \
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>  \
    inline bool                                                                \
    all_##NAME(tiny_vector<V1, N1, R1> const & l,                              \
               tiny_vector<V2, N2, R2> const & r)                              \
    {                                                                          \
        XVIGRA_ASSERT_MSG(l.size() == r.size(),                                \
            "tiny_vector::all_" #NAME "(): size mismatch.");                   \
        for(decltype(l.size()) k=0; k < l.size(); ++k)                         \
            if (l[k] OP r[k])                                                  \
                return false;                                                  \
        return true;                                                           \
    }                                                                          \
                                                                               \
    template <class V1, index_t N1, class R1, class V2,                        \
              VIGRA_REQUIRE<!tiny_vector_concept<V2>::value &&                 \
                              std::is_convertible<V2, V1>::value> >            \
    inline bool                                                                \
    all_##NAME(tiny_vector<V1, N1, R1> const & l,                              \
               V2 const & r)                                                   \
    {                                                                          \
        for(decltype(l.size()) k=0; k < l.size(); ++k)                         \
            if (l[k] OP r)                                                     \
                return false;                                                  \
        return true;                                                           \
    }                                                                          \
                                                                               \
    template <class V1, class V2, index_t N2, class R2,                        \
              VIGRA_REQUIRE<!tiny_vector_concept<V1>::value &&                 \
                              std::is_convertible<V2, V1>::value> >            \
    inline bool                                                                \
    all_##NAME(V1 const & l,                                                   \
               tiny_vector<V2, N2, R2> const & r)                              \
    {                                                                          \
        for(decltype(r.size()) k=0; k < r.size(); ++k)                         \
            if (l OP r[k])                                                     \
                return false;                                                  \
        return true;                                                           \
    }

    XVIGRA_TINY_COMPARISON(less, >=)
    XVIGRA_TINY_COMPARISON(less_equal, >)
    XVIGRA_TINY_COMPARISON(greater, <=)
    XVIGRA_TINY_COMPARISON(greater_equal, <)

    #undef XVIGRA_TINY_COMPARISON

    template <class V, index_t N1, class R1, index_t N2, class R2>
    inline bool
    all_close(tiny_vector<V, N1, R1> const & l,
              tiny_vector<V, N2, R2> const & r,
              double rtol = default_tolerance<V>::value,
              double atol = default_tolerance<V>::value,
              bool equal_nan = false)
    {
        if(l.size() != r.size())
            return false;
         for(decltype(l.size()) k=0; k < l.size(); ++k)
            if(!is_close(l[k], r[k], rtol, atol, equal_nan))
                return false;
        return true;
    }

    template <class V, index_t N1, class R1, index_t N2, class R2>
    inline bool
    all_close(tiny_vector<V, N1, R1> const & l,
              V const & r,
              double rtol = 2.0*std::numeric_limits<double>::epsilon(),
              double atol = 2.0*std::numeric_limits<double>::epsilon(),
              bool equal_nan = false)
    {
         for(decltype(l.size()) k=0; k < l.size(); ++k)
            if(!is_close(l[k], r, rtol, atol, equal_nan))
                return false;
        return true;
    }

    template <class V, index_t N1, class R1, index_t N2, class R2>
    inline bool
    all_close(V const & l,
              tiny_vector<V, N2, R2> const & r,
              double rtol = 2.0*std::numeric_limits<double>::epsilon(),
              double atol = 2.0*std::numeric_limits<double>::epsilon(),
              bool equal_nan = false)
    {
         for(decltype(r.size()) k=0; k < r.size(); ++k)
            if(!is_close(l, r[k], rtol, atol, equal_nan))
                return false;
        return true;
    }

    /******************************/
    /* tiny_vector implementation */
    /******************************/

    template <class V, index_t N, class R>
    inline
    tiny_vector<V, N, R>::tiny_vector()
    : base_type()
    {
    }

    template <class V, index_t N, class R>
    inline
    tiny_vector<V, N, R>::tiny_vector(tiny_vector const & v)
    : base_type(v)
    {
    }

    template <class V, index_t N, class R>
    inline
    tiny_vector<V, N, R>::tiny_vector(tiny_vector && v)
    : base_type(std::forward<tiny_vector>(v))
    {
    }

    template <class V, index_t N, class R>
    template <class T, index_t M, class Q>
    inline
    tiny_vector<V, N, R>::tiny_vector(tiny_vector<T, M, Q> const & v)
    : base_type(v.cbegin(), v.cend())
    {
    }

    template <class V, index_t N, class R>
    template <class T, class A>
    inline
    tiny_vector<V, N, R>::tiny_vector(std::vector<T, A> const & v)
    : base_type(v.cbegin(), v.cend())
    {
    }

    template <class V, index_t N, class R>
    template <class T, std::size_t M>
    inline
    tiny_vector<V, N, R>::tiny_vector(std::array<T, M> const & v)
    : base_type(v.cbegin(), v.cend())
    {
    }

    template <class V, index_t N, class R>
    template <class T, std::size_t M, class A, bool I>
    inline
    tiny_vector<V, N, R>::tiny_vector(xt::svector<T, M, A, I> const & v)
    : base_type(v.cbegin(), v.cend())
    {
    }

    template <class V, index_t N, class R>
    template <class T, class A>
    inline
    tiny_vector<V, N, R>::tiny_vector(xt::uvector<T, A> const & v)
    : base_type(v.cbegin(), v.cend())
    {
    }

    template <class V, index_t N, class R>
    inline auto
    tiny_vector<V, N, R>::operator=(tiny_vector const & v) -> tiny_vector &
    {
        base_type::operator=(v);
        return *this;
    }

    template <class V, index_t N, class R>
    inline auto
    tiny_vector<V, N, R>::operator=(tiny_vector && v) -> tiny_vector &
    {
        base_type::operator=(v);
        return *this;
    }

    template <class V, index_t N, class R>
    inline auto
    tiny_vector<V, N, R>::operator=(value_type const & v) -> tiny_vector &
    {
        base_type::assign(size(), v);
        return *this;
    }

    template <class V, index_t N, class R>
    template <class T, class A>
    inline auto
    tiny_vector<V, N, R>::operator=(std::vector<T, A> const & v) -> tiny_vector &
    {
        base_type::assign(v.cbegin(), v.cend());
        return *this;
    }

    template <class V, index_t N, class R>
    template <class T, std::size_t M>
    inline auto
    tiny_vector<V, N, R>::operator=(std::array<T, M> const & v) -> tiny_vector &
    {
        base_type::assign(v.cbegin(), v.cend());
        return *this;
    }

    template <class V, index_t N, class R>
    template <class T, std::size_t M, class A, bool I>
    inline auto
    tiny_vector<V, N, R>::operator=(xt::svector<T, M, A, I> const & v) -> tiny_vector &
    {
        base_type::assign(v.cbegin(), v.cend());
        return *this;
    }

    template <class V, index_t N, class R>
    template <class T, class A>
    inline auto
    tiny_vector<V, N, R>::operator=(xt::uvector<T, A> const & v) -> tiny_vector &
    {
        base_type::assign(v.cbegin(), v.cend());
        return *this;
    }

    template <class V, index_t N, class R>
    template <class U, index_t M, class Q>
    inline auto
    tiny_vector<V, N, R>::operator=(tiny_vector<U, M, Q> const & v) -> tiny_vector &
    {
        base_type::assign(v.cbegin(), v.cend());
        return *this;
    }

    template <class V, index_t N, class R>
    inline void
    tiny_vector<V, N, R>::assign(std::initializer_list<value_type> v)
    {
        base_type::assign(v.begin(), v.end());
    }

    template <class V, index_t N, class R>
    inline auto
    tiny_vector<V, N, R>::at(size_type i) -> reference
    {
        if(i < 0 || i >= size())
        {
            throw std::out_of_range("tiny_vector::at()");
        }
        return (*this)[i];
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    tiny_vector<V, N, R>::at(size_type i) const -> const_reference
    {
        if(i < 0 || i >= size())
        {
            throw std::out_of_range("tiny_vector::at()");
        }
        return (*this)[i];
    }

    template <class V, index_t N, class R>
    inline auto
    tiny_vector<V, N, R>::front() -> reference
    {
        return (*this)[0];
    }

    template <class V, index_t N, class R>
    inline auto
    tiny_vector<V, N, R>::back() -> reference
    {
        return (*this)[size()-1];
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    tiny_vector<V, N, R>::front() const -> const_reference
    {
        return (*this)[0];
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    tiny_vector<V, N, R>::back() const -> const_reference
    {
        return (*this)[size()-1];
    }

    template <class V, index_t N, class R>
    template <index_t FROM, index_t TO>
    inline auto
    tiny_vector<V, N, R>::subarray()
    {
        static_assert(FROM >= 0 && FROM < TO,
            "tiny_vector::subarray(): range out of bounds.");
        vigra_precondition(TO <= size(),
            "tiny_vector::subarray(): range out of bounds.");
        return tiny_vector<value_type, TO-FROM, iterator>(begin()+FROM);
    }

    template <class V, index_t N, class R>
    template <index_t FROM, index_t TO>
    inline auto
    tiny_vector<V, N, R>::subarray() const
    {
        static_assert(FROM >= 0 && FROM < TO,
            "tiny_vector::subarray(): range out of bounds.");
        vigra_precondition(TO <= size(),
            "tiny_vector::subarray(): range out of bounds.");
        return tiny_vector<const_value_type, TO-FROM, const_iterator>(begin()+FROM);
    }

    template <class V, index_t N, class R>
    inline auto
    tiny_vector<V, N, R>::subarray(size_type FROM, size_type TO)
    {
        vigra_precondition(FROM >= 0 && FROM < TO && TO <= size(),
            "tiny_vector::subarray(): range out of bounds.");
        return tiny_vector<value_type, runtime_size, iterator>(begin()+FROM, begin()+TO);
    }

    template <class V, index_t N, class R>
    inline auto
    tiny_vector<V, N, R>::subarray(size_type FROM, size_type TO) const
    {
        vigra_precondition(FROM >= 0 && FROM < TO && TO <= size(),
            "tiny_vector::subarray(): range out of bounds.");
        return tiny_vector<const_value_type, runtime_size, const_iterator>(begin()+FROM, begin()+TO);
    }

    template <class V, index_t N, class R>
    inline auto
    tiny_vector<V, N, R>::erase(size_type m) const
    {
        vigra_precondition(m >= 0 && m < size(), "tiny_vector::erase(): "
            "Index "+std::to_string(m)+" out of bounds [0, "+std::to_string(size())+").");
        constexpr static index_t res_size = has_fixed_size
                                            ? static_size-1
                                            : runtime_size;
        tiny_vector<value_type, res_size> res(size()-1, dont_init);
        std::copy(cbegin(), cbegin()+m, res.begin());
        std::copy(cbegin()+m+1, cend(), res.begin()+m);
        return res;
    }

    template <class V, index_t N, class R>
    inline auto
    tiny_vector<V, N, R>::pop_front() const
    {
        return erase(0);
    }

    template <class V, index_t N, class R>
    inline auto
    tiny_vector<V, N, R>::pop_back() const
    {
        return erase(size()-1);
    }

    template <class V, index_t N, class R>
    inline auto
    tiny_vector<V, N, R>::insert(size_type m, value_type v) const
    {
        vigra_precondition(m >= 0 && m <= size(), "tiny_vector::insert(): "
            "Index "+std::to_string(m)+" out of bounds [0, "+std::to_string(size())+"].");
        constexpr static index_t res_size = has_fixed_size
                                            ? static_size+1
                                            : runtime_size;
        tiny_vector<value_type, res_size> res(size()+1, dont_init);
        std::copy(cbegin(), cbegin()+m, res.begin());
        res[m] = v;
        std::copy(cbegin()+m, cend(), res.begin()+m+1);
        return res;
    }

    template <class V, index_t N, class R>
    inline auto
    tiny_vector<V, N, R>::push_front(value_type v) const
    {
        return insert(0, v);
    }

    template <class V, index_t N, class R>
    inline auto
    tiny_vector<V, N, R>::push_back(value_type v) const
    {
        return insert(size(), v);
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    tiny_vector<V, N, R>::cbegin() const -> const_iterator
    {
        return base_type::begin();
    }

    template <class V, index_t N, class R>
    inline auto
    tiny_vector<V, N, R>::end() -> iterator
    {
        return begin() + static_cast<std::ptrdiff_t>(size());
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    tiny_vector<V, N, R>::end() const -> const_iterator
    {
        return begin() + static_cast<std::ptrdiff_t>(size());
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    tiny_vector<V, N, R>::cend() const -> const_iterator
    {
        return cbegin() + static_cast<std::ptrdiff_t>(size());
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    tiny_vector<V, N, R>::crbegin() const -> const_reverse_iterator
    {
        return base_type::rbegin();
    }

    template <class V, index_t N, class R>
    inline auto
    tiny_vector<V, N, R>::rend() -> reverse_iterator
    {
        return rbegin() + static_cast<std::ptrdiff_t>(size());
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    tiny_vector<V, N, R>::rend() const -> const_reverse_iterator
    {
        return rbegin() + static_cast<std::ptrdiff_t>(size());
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    tiny_vector<V, N, R>::crend() const -> const_reverse_iterator
    {
        return crbegin() + static_cast<std::ptrdiff_t>(size());
    }

    template <class V, index_t N, class R>
    constexpr bool
    tiny_vector<V, N, R>::empty() const
    {
        return size() == 0;
    }

    template <class V, index_t N, class R>
    template <index_t SIZE>
    auto tiny_vector<V, N, R>::unit_vector(index_t k)
    {
        static_assert(SIZE > 0,
            "tiny_vector::unit_vector(): SIZE must be poisitive.");
        tiny_vector<V, SIZE> res(SIZE);
        res[k] = 1;
        return res;
    }

        /// factory function for the fixed-size k-th unit vector
    template <class V, index_t N, class R>
    auto tiny_vector<V, N, R>::unit_vector(index_t size, index_t k)
    {
        tiny_vector<V, runtime_size> res(size);
        res[k] = 1;
        return res;
    }

        /// factory function for fixed-size linear sequence ending at <tt>end-1</tt>
    template <class V, index_t N, class R>
    auto tiny_vector<V, N, R>::range(value_type end)
    {
        vigra_precondition(static_size != runtime_size || end >= 0,
            "tiny_vector::range(): end must be non-negative.");
        auto start = (static_size != runtime_size)
                        ? end - static_cast<value_type>(static_size)
                        : value_type();
        tiny_vector<value_type, N> res(end-start, dont_init);
        for(decltype(res.size()) k=0; k < res.size(); ++k, ++start)
            res[k] = start;
        return res;
    }

        /// factory function for a linear sequence from <tt>begin</tt> to <tt>end</tt>
        /// (exclusive) with stepsize <tt>step</tt>
    template <class V, index_t N, class R>
    template <class T1, class T2>
    auto tiny_vector<V, N, R>::range(value_type begin_, T1 end_, T2 step_)
    {
        using namespace math;
        using T = promote_type_t<value_type, T1, T2>;
        T begin = begin_,
          end   = end_,
          step  = step_;
        vigra_precondition(step != 0,
            "tiny_vector::range(): step must be non-zero.");
        vigra_precondition((step > 0 && begin <= end) || (step < 0 && begin >= end),
            "tiny_vector::range(): sign mismatch between step and (end-begin).");
        // use floor() here because value_type could be floating point
        index_t size = (index_t)floor((abs(end-begin+step)-1)/abs(step));
        tiny_vector<value_type, runtime_size> res(size, dont_init);
        for(index_t k=0; k < size; ++k, begin += step)
            res[k] = static_cast<V>(begin);
        return res;
    }

    /*************************************************/
    /* tiny_vector_impl dynamic shape implementation */
    /*************************************************/

    template <class V, index_t B>
    inline
    tiny_vector_impl<V, runtime_size, V[B]>::tiny_vector_impl()
    : m_size(0)
    , m_data(m_buffer)
    {
    }

    template <class V, index_t B>
    inline
    tiny_vector_impl<V, runtime_size, V[B]>::~tiny_vector_impl()
    {
        deallocate();
    }

    template <class V, index_t B>
    inline
    tiny_vector_impl<V, runtime_size, V[B]>::tiny_vector_impl(size_type n)
    : m_size(n)
    , m_data(m_buffer)
    {
        allocate();
    }

    template <class V, index_t B>
    inline
    tiny_vector_impl<V, runtime_size, V[B]>::tiny_vector_impl(size_type n, const value_type& v)
    : m_size(n)
    , m_data(m_buffer)
    {
        allocate(v);
    }

    template <class V, index_t B>
    inline
    tiny_vector_impl<V, runtime_size, V[B]>::tiny_vector_impl(size_type n, tags::skip_initialization_tag)
    : m_size(n)
    , m_data(m_buffer)
    {
        allocate(dont_init);
    }

    template <class V, index_t B>
    template <class IT, class>
    inline
    tiny_vector_impl<V, runtime_size, V[B]>::tiny_vector_impl(IT begin, IT end)
    : m_size(0)
    , m_data(m_buffer)
    {
        assign(begin, end);
    }

    template <class V, index_t B>
    inline
    tiny_vector_impl<V, runtime_size, V[B]>::tiny_vector_impl(std::initializer_list<value_type> const & v)
    : m_size(0)
    , m_data(m_buffer)
    {
        assign(v.begin(), v.end());
    }

    template <class V, index_t B>
    inline
    tiny_vector_impl<V, runtime_size, V[B]>::tiny_vector_impl(tiny_vector_impl const & v)
    : tiny_vector_impl(v.begin(), v.begin()+v.size())
    {
    }

    template <class V, index_t B>
    inline
    tiny_vector_impl<V, runtime_size, V[B]>::tiny_vector_impl(tiny_vector_impl && v)
    : m_size(0)
    , m_data(m_buffer)
    {
        v.swap(*this);
    }

    template <class V, index_t B>
    inline auto
    tiny_vector_impl<V, runtime_size, V[B]>::operator=(tiny_vector_impl const & v) -> tiny_vector_impl &
    {
        if(this != &v)
        {
            assign(v.begin(), v.begin()+v.size());
        }
        return *this;
    }

    template <class V, index_t B>
    inline auto
    tiny_vector_impl<V, runtime_size, V[B]>::operator=(tiny_vector_impl && v) -> tiny_vector_impl &
    {
        if(this != &v)
        {
            assign(v.begin(), v.begin() + v.size());
        }
        return *this;
    }

    template <class V, index_t B>
    inline void
    tiny_vector_impl<V, runtime_size, V[B]>::assign(size_type n, const value_type& v)
    {
        if(m_size == n)
        {
            std::fill(begin(), begin()+size(), v);
        }
        else
        {
            deallocate();
            m_size = n;
            allocate(v);
        }
    }

    template <class V, index_t B>
    template <class IT, class>
    inline void
    tiny_vector_impl<V, runtime_size, V[B]>::assign(IT begin, IT end)
    {
        size_type n = static_cast<size_type>(std::distance(begin, end));
        if(m_size == n)
        {
            for (size_type k = 0; k < m_size; ++k, ++begin)
            {
                m_data[k] = static_cast<value_type>(*begin);
            }
        }
        else
        {
            deallocate();
            m_size = n;
            allocate(begin);
        }
    }

    template <class V, index_t B>
    inline auto
    tiny_vector_impl<V, runtime_size, V[B]>::data() -> pointer
    {
        return m_data;
    }

    template <class V, index_t B>
    constexpr inline auto
    tiny_vector_impl<V, runtime_size, V[B]>::data() const -> const_pointer
    {
        return m_data;
    }

    template <class V, index_t B>
    inline auto
    tiny_vector_impl<V, runtime_size, V[B]>::operator[](size_type i) -> reference
    {
        return m_data[i];
    }

    template <class V, index_t B>
    constexpr inline auto
    tiny_vector_impl<V, runtime_size, V[B]>::operator[](size_type i) const -> const_reference
    {
        return m_data[i];
    }

    template <class V, index_t B>
    inline void
    tiny_vector_impl<V, runtime_size, V[B]>::resize(size_type n)
    {
        if(n != m_size)
        {
            deallocate();
            m_size = n;
            allocate();
        }
    }

    template <class V, index_t B>
    inline auto
    tiny_vector_impl<V, runtime_size, V[B]>::capacity() const -> size_type
    {
        return std::max<std::size_t>(m_size, buffer_size);
    }

    template <class V, index_t B>
    inline auto
    tiny_vector_impl<V, runtime_size, V[B]>::size() const -> size_type
    {
        return m_size;
    }

    template <class V, index_t B>
    inline auto
    tiny_vector_impl<V, runtime_size, V[B]>::max_size() const -> size_type
    {
        return m_allocator.max_size();
    }

    template <class V, index_t B>
    inline bool
    tiny_vector_impl<V, runtime_size, V[B]>::on_stack() const
    {
        return m_data == m_buffer;
    }

    template <class V, index_t B>
    inline auto
    tiny_vector_impl<V, runtime_size, V[B]>::begin() -> iterator
    {
        return m_data;
    }

    template <class V, index_t B>
    inline auto
    tiny_vector_impl<V, runtime_size, V[B]>::begin() const -> const_iterator
    {
        return m_data;
    }

    template <class V, index_t B>
    inline auto
    tiny_vector_impl<V, runtime_size, V[B]>::rbegin() -> reverse_iterator
    {
        return reverse_iterator(m_data + m_size);
    }

    template <class V, index_t B>
    inline auto
    tiny_vector_impl<V, runtime_size, V[B]>::rbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(m_data + m_size);
    }

    template <class V, index_t B>
    inline void
    tiny_vector_impl<V, runtime_size, V[B]>::swap(tiny_vector_impl & other)
    {
        using std::swap;
        if(this == &other)
        {
            return;
        }
        if(m_size == 0 || m_size > buffer_size)
        {
            if(other.m_size == 0 || other.m_size > buffer_size)
            {
                // both use allocated memory (or no memory at all)
                swap(m_data, other.m_data);
            }
            else
            {
                // self uses allocated memory, other the buffer
                for(size_type k=0; k<other.m_size; ++k)
                {
                    m_buffer[k] = other.m_data[k];
                }
                other.m_data = m_data;
                m_data = m_buffer;
            }
        }
        else
        {
            if(other.m_size > buffer_size)
            {
                // self uses the buffer, other allocated memory
                for(size_type k=0; k<m_size; ++k)
                {
                    other.m_buffer[k] = m_data[k];
                }
                m_data = other.m_data;
                other.m_data = other.m_buffer;
            }
            else
            {
                // both use the buffer
                if(m_size < other.m_size)
                {
                    for(size_type k=0; k<m_size; ++k)
                    {
                        swap(m_data[k], other.m_data[k]);
                    }
                    for(size_type k=m_size; k<other.m_size; ++k)
                    {
                        m_data[k] = other.m_data[k];
                    }
                }
                else
                {
                    for(size_type k=0; k<other.m_size; ++k)
                    {
                        swap(m_data[k], other.m_data[k]);
                    }
                    for(size_type k=other.m_size; k<m_size; ++k)
                    {
                        other.m_data[k] = m_data[k];
                    }
                }
            }
        }
        swap(m_size, other.m_size);
    }

    template <class V, index_t B>
    inline void
    tiny_vector_impl<V, runtime_size, V[B]>::allocate(value_type const & v)
    {
        if(m_size > buffer_size)
        {
            m_data = m_allocator.allocate(m_size);
            std::uninitialized_fill(m_data, m_data+m_size, v);
        }
        else
        {
            std::fill(m_data, m_data+m_size, v);
        }
    }

    template <class V, index_t B>
    inline void
    tiny_vector_impl<V, runtime_size, V[B]>::allocate(tags::skip_initialization_tag)
    {
        if(m_size > buffer_size)
        {
            m_data = m_allocator.allocate(m_size);
            if(!may_use_uninitialized_memory)
            {
                std::uninitialized_fill(m_data, m_data+m_size, value_type());
            }
        }
    }

    template <class V, index_t B>
    template <class IT, class>
    inline void
    tiny_vector_impl<V, runtime_size, V[B]>::allocate(IT begin)
    {
        if(m_size > buffer_size)
        {
            m_data = m_allocator.allocate(m_size);
            for(size_type k=0; k<m_size; ++k, ++begin)
            {
                m_allocator.construct(m_data+k, static_cast<value_type>(*begin));
            }
        }
        else
        {
            for(size_type k=0; k<m_size; ++k, ++begin)
            {
                m_data[k] = static_cast<value_type>(*begin);
            }
        }
    }

    template <class V, index_t B>
    inline void
    tiny_vector_impl<V, runtime_size, V[B]>::deallocate()
    {
        if(m_size > buffer_size)
        {
            if(!may_use_uninitialized_memory)
            {
                for(size_type k=0; k<m_size; ++k)
                {
                    m_allocator.destroy(m_data+k);
                }
            }
            m_allocator.deallocate(m_data, m_size);
            m_data = m_buffer;
        }
        m_size = 0;
    }

    /***********************************************/
    /* tiny_vector_impl fixed shape implementation */
    /***********************************************/

    template <class V, index_t N>
    inline
    tiny_vector_impl<V, N, std::array<V, (size_t)N>>::tiny_vector_impl()
    : base_type{}
    {
    }

    template <class V, index_t N>
    inline
    tiny_vector_impl<V, N, std::array<V, (size_t)N>>::tiny_vector_impl(size_type n)
    : tiny_vector_impl()
    {
        std::ignore = n;
        XVIGRA_ASSERT_MSG(n == size(), "tiny_vector_impl(n): size mismatch");
    }

    template <class V, index_t N>
    inline
    tiny_vector_impl<V, N, std::array<V, (size_t)N>>::tiny_vector_impl(tags::skip_initialization_tag)
    {}

    template <class V, index_t N>
    inline
    tiny_vector_impl<V, N, std::array<V, (size_t)N>>::tiny_vector_impl(size_type n, const value_type& v)
    {
        std::ignore = n;
        XVIGRA_ASSERT_MSG(n == size(), "tiny_vector_impl(n): size mismatch");
        base_type::fill(v);
    }

    template <class V, index_t N>
    inline
    tiny_vector_impl<V, N, std::array<V, (size_t)N>>::tiny_vector_impl(size_type n, tags::skip_initialization_tag)
    {
        std::ignore = n;
        XVIGRA_ASSERT_MSG(n == size(), "tiny_vector_impl(n): size mismatch.");
    }

    template <class V, index_t N>
    template <class IT, class>
    inline
    tiny_vector_impl<V, N, std::array<V, (size_t)N>>::tiny_vector_impl(IT begin)
    {
        assign(begin, begin+N);
    }

    template <class V, index_t N>
    template <class IT, class>
    inline
    tiny_vector_impl<V, N, std::array<V, (size_t)N>>::tiny_vector_impl(IT begin, IT end)
    {
        assign(begin, end);
    }

    template <class V, index_t N>
    inline
    tiny_vector_impl<V, N, std::array<V, (size_t)N>>::tiny_vector_impl(std::initializer_list<value_type> const & v)
    {
        const size_t n = v.size();
        if(n == 1)
        {
            assign(N, static_cast<value_type>(*v.begin()));
        }
        else if(n == N)
        {
            assign(v.begin(), v.end());
        }
        else
        {
            XVIGRA_ASSERT_MSG(false, "tiny_vector_impl::tiny_vector_impl(std::initializer_list): size mismatch.");
        }
    }

    template <class V, index_t N>
    inline
    tiny_vector_impl<V, N, std::array<V, (size_t)N>>::tiny_vector_impl(tiny_vector_impl const & v)
    : base_type(v)
    {
    }

    template <class V, index_t N>
    inline
    tiny_vector_impl<V, N, std::array<V, (size_t)N>>::tiny_vector_impl(tiny_vector_impl && v)
    : base_type(std::forward<base_type>(v))
    {
    }

    template <class V, index_t N>
    inline auto
    tiny_vector_impl<V, N, std::array<V, (size_t)N>>::operator=(tiny_vector_impl const & v) -> tiny_vector_impl &
    {
        base_type::operator=(v);
        return *this;
    }

    template <class V, index_t N>
    inline auto
    tiny_vector_impl<V, N, std::array<V, (size_t)N>>::operator=(tiny_vector_impl && v) -> tiny_vector_impl &
    {
        base_type::operator=(std::forward<base_type>(v));
        return *this;
    }

    template <class V, index_t N>
    inline void
    tiny_vector_impl<V, N, std::array<V, (size_t)N>>::assign(size_type n, const value_type& v)
    {
        std::ignore = n;
        XVIGRA_ASSERT_MSG(n == size(), "tiny_vector_impl::assign(n, v): size mismatch.");
        base_type::fill(v);
    }

    template <class V, index_t N>
    inline void
    tiny_vector_impl<V, N, std::array<V, (size_t)N>>::resize(std::size_t s)
    {
        vigra_precondition(s == this->size(),
            "tiny_vector::resize(): size mismatch.");
    }

    template <class V, index_t N>
    template <class IT, class>
    inline void
    tiny_vector_impl<V, N, std::array<V, (size_t)N>>::assign(IT begin, IT end)
    {
        std::ignore = end;
        XVIGRA_ASSERT_MSG(std::distance(begin, end) == static_cast<std::ptrdiff_t>(size()),
            "tiny_vector_impl::assign(begin, end): size mismatch.");
        for(size_type k=0; k<N; ++k, ++begin)
        {
            (*this)[k] = static_cast<value_type>(*begin);
        }
    }

    template <class V, index_t N>
    constexpr inline auto
    tiny_vector_impl<V, N, std::array<V, (size_t)N>>::capacity() const -> size_type
    {
        return N;
    }

    /****************************************************/
    /* tiny_vector_impl fixed shape view implementation */
    /****************************************************/

    template <class V, index_t N, class R>
    inline
    tiny_vector_impl<V, N, R>::tiny_vector_impl()
    : m_data()
    {
    }

    template <class V, index_t N, class R>
    inline
    tiny_vector_impl<V, N, R>::tiny_vector_impl(representation_type const & begin)
    : m_data(begin)
    {
    }

    template <class V, index_t N, class R>
    inline
    tiny_vector_impl<V, N, R>::tiny_vector_impl(representation_type const & begin, representation_type const & end)
    : m_data(begin)
    {
        std::ignore = end;
        XVIGRA_ASSERT_MSG(std::distance(begin, end) == static_cast<std::ptrdiff_t>(size()),
            "tiny_vector_impl(begin, end): size mismatch");
    }

    template <class V, index_t N, class R>
    template <class IT, class>
    inline
    tiny_vector_impl<V, N, R>::tiny_vector_impl(IT begin, IT end)
    : m_data(const_cast<representation_type>(&*begin))
    {
        std::ignore = end;
        XVIGRA_ASSERT_MSG(std::distance(begin, end) == static_cast<std::ptrdiff_t>(size()),
            "tiny_vector_impl::assign(begin, end): size mismatch.");
    }

    template <class V, index_t N, class R>
    inline void
    tiny_vector_impl<V, N, R>::reset(representation_type const & begin)
    {
        m_data = begin;
    }

    template <class V, index_t N, class R>
    inline void
    tiny_vector_impl<V, N, R>::resize(std::size_t s)
    {
        vigra_precondition(s == this->size(),
            "tiny_vector::resize(): size mismatch.");
    }

    template <class V, index_t N, class R>
    inline void
    tiny_vector_impl<V, N, R>::assign(size_type n, const value_type& v)
    {
        std::ignore = n;
        XVIGRA_ASSERT_MSG(n == size(), "tiny_vector_impl::assign(n, v): size mismatch.");
        for(size_type k=0; k<N; ++k)
        {
            (*this)[k] = v;
        }
    }

    template <class V, index_t N, class R>
    template <class IT, class>
    inline void
    tiny_vector_impl<V, N, R>::assign(IT begin, IT end)
    {
        std::ignore = end;
        XVIGRA_ASSERT_MSG(std::distance(begin, end) == static_cast<std::ptrdiff_t>(size()),
            "tiny_vector_impl::assign(begin, end): size mismatch.");
        for(size_type k=0; k<N; ++k, ++begin)
        {
            (*this)[k] = static_cast<value_type>(*begin);
        }
    }

    template <class V, index_t N, class R>
    inline auto
    tiny_vector_impl<V, N, R>::operator[](size_type i) -> reference
    {
        return m_data[i];
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    tiny_vector_impl<V, N, R>::operator[](size_type i) const -> const_reference
    {
        return m_data[i];
    }

    template <class V, index_t N, class R>
    inline auto
    tiny_vector_impl<V, N, R>::data() -> pointer
    {
        return &m_data[0];
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    tiny_vector_impl<V, N, R>::data() const -> const_pointer
    {
        return &m_data[0];
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    tiny_vector_impl<V, N, R>::size() const -> size_type
    {
        return N;
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    tiny_vector_impl<V, N, R>::max_size() const -> size_type
    {
        return N;
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    tiny_vector_impl<V, N, R>::capacity() const -> size_type
    {
        return N;
    }

    template <class V, index_t N, class R>
    inline auto
    tiny_vector_impl<V, N, R>::begin() -> iterator
    {
        return m_data;
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    tiny_vector_impl<V, N, R>::begin() const -> const_iterator
    {
        return m_data;
    }

    template <class V, index_t N, class R>
    inline auto
    tiny_vector_impl<V, N, R>::rbegin() -> reverse_iterator
    {
        return reverse_iterator(m_data+N);
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    tiny_vector_impl<V, N, R>::rbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(m_data+N);
    }

    template <class V, index_t N, class R>
    inline void
    tiny_vector_impl<V, N, R>::swap(tiny_vector_impl & other)
    {
        using std::swap;
        swap(m_data, other.m_data);
    }

    /******************************************************/
    /* tiny_vector_impl dynamic shape view implementation */
    /******************************************************/

    template <class V, class R>
    inline
    tiny_vector_impl<V, runtime_size, R>::tiny_vector_impl()
    : m_size(0)
    , m_data()
    {
    }

    template <class V, class R>
    inline
    tiny_vector_impl<V, runtime_size, R>::tiny_vector_impl(representation_type const & begin, representation_type const & end)
    : m_size(static_cast<size_type>(std::distance(begin, end)))
    , m_data(begin)
    {
    }

    template <class V, class R>
    template <class IT, class>
    inline
    tiny_vector_impl<V, runtime_size, R>::tiny_vector_impl(IT begin, IT end)
    : m_size(static_cast<size_type>(std::distance(begin, end)))
    , m_data(const_cast<representation_type>(&*begin))
    {
    }

    template <class V, class R>
    inline void
    tiny_vector_impl<V, runtime_size, R>::reset(representation_type const & begin, representation_type const & end)
    {
        m_size = static_cast<size_type>(std::distance(begin, end));
        m_data = begin;
    }

    template <class V, class R>
    inline void
    tiny_vector_impl<V, runtime_size, R>::resize(std::size_t s)
    {
        vigra_precondition(s == this->size(),
            "tiny_vector::resize(): size mismatch.");
    }

    template <class V, class R>
    inline void
    tiny_vector_impl<V, runtime_size, R>::assign(size_type n, const value_type& v)
    {
        std::ignore = n;
        XVIGRA_ASSERT_MSG(n == size(), "tiny_vector_impl::assign(n, v): size mismatch.");
        std::fill(begin(), begin()+size(), v);
    }

    template <class V, class R>
    template <class IT, class>
    inline void
    tiny_vector_impl<V, runtime_size, R>::assign(IT begin, IT end)
    {
        std::ignore = end;
        XVIGRA_ASSERT_MSG(std::distance(begin, end) == static_cast<std::ptrdiff_t>(size()),
            "tiny_vector_impl::assign(begin, end): size mismatch.");
        for(size_type k=0; k<size(); ++k, ++begin)
        {
            (*this)[k] = static_cast<value_type>(*begin);
        }
    }

    template <class V, class R>
    inline auto
    tiny_vector_impl<V, runtime_size, R>::operator[](size_type i) -> reference
    {
        return m_data[i];
    }

    template <class V, class R>
    constexpr inline auto
    tiny_vector_impl<V, runtime_size, R>::operator[](size_type i) const -> const_reference
    {
        return m_data[i];
    }

    template <class V, class R>
    inline auto
    tiny_vector_impl<V, runtime_size, R>::data() -> pointer
    {
        return &m_data[0];
    }

    template <class V, class R>
    constexpr inline auto
    tiny_vector_impl<V, runtime_size, R>::data() const -> const_pointer
    {
        return &m_data[0];
    }

    template <class V, class R>
    constexpr inline auto
    tiny_vector_impl<V, runtime_size, R>::size() const -> size_type
    {
        return m_size;
    }

    template <class V, class R>
    constexpr inline auto
    tiny_vector_impl<V, runtime_size, R>::max_size() const -> size_type
    {
        return m_size;
    }

    template <class V, class R>
    constexpr inline auto
    tiny_vector_impl<V, runtime_size, R>::capacity() const -> size_type
    {
        return m_size;
    }

    template <class V, class R>
    inline auto
    tiny_vector_impl<V, runtime_size, R>::begin() -> iterator
    {
        return m_data;
    }

    template <class V, class R>
    constexpr inline auto
    tiny_vector_impl<V, runtime_size, R>::begin() const -> const_iterator
    {
        return m_data;
    }

    template <class V, class R>
    inline auto
    tiny_vector_impl<V, runtime_size, R>::rbegin() -> reverse_iterator
    {
        return reverse_iterator(m_data+m_size);
    }

    template <class V, class R>
    constexpr inline auto
    tiny_vector_impl<V, runtime_size, R>::rbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(m_data+m_size);
    }

    template <class V, class R>
    inline void
    tiny_vector_impl<V, runtime_size, R>::swap(tiny_vector_impl & other)
    {
        using std::swap;
        swap(m_size, other.m_size);
        swap(m_data, other.m_data);
    }

    /************************************************************/
    /* tiny_vector_impl xt::xbuffer_adaptor view implementation */
    /************************************************************/

    template <class V, class CP, class O, class A>
    inline void
    tiny_vector_impl<V, runtime_size, xt::xbuffer_adaptor<CP, O, A>>::assign(size_type n, const value_type& v)
    {
        std::ignore = n;
        XVIGRA_ASSERT_MSG(n == size(), "tiny_vector_impl::assign(n, v): size mismatch.");
        std::fill(begin(), begin()+size(), v);
    }

    template <class V, class CP, class O, class A>
    template <class IT, class>
    inline void
    tiny_vector_impl<V, runtime_size, xt::xbuffer_adaptor<CP, O, A>>::assign(IT begin, IT end)
    {
        std::ignore = end;
        XVIGRA_ASSERT_MSG(std::distance(begin, end) == static_cast<std::ptrdiff_t>(size()),
            "tiny_vector_impl::assign(begin, end): size mismatch.");
        for(size_type k=0; k<size(); ++k, ++begin)
        {
            (*this)[k] = static_cast<value_type>(*begin);
        }
    }

    template <class V, class CP, class O, class A>
    constexpr inline auto
    tiny_vector_impl<V, runtime_size, xt::xbuffer_adaptor<CP, O, A>>::max_size() const -> size_type
    {
        return size();
    }

    template <class V, class CP, class O, class A>
    constexpr inline auto
    tiny_vector_impl<V, runtime_size, xt::xbuffer_adaptor<CP, O, A>>::capacity() const -> size_type
    {
        return size();
    }

    /*************************/
    /* tiny_vector factories */
    /*************************/

    template <class V, index_t N, class R, class W = V>
    inline auto
    unit_vector(tiny_vector<V, N, R> const & tmpl, index_t axis, W const & w = W(1))
    {
        tiny_vector<V, N> res(tmpl.size(), 0);
        res[axis] = w;
        return res;
    }

        /// \brief compute the F-order or C-order (default) stride of a given shape.
        /// Example: {200, 100, 50}  =>  {5000, 50, 1}
    template <class V, index_t N, class R>
    inline auto
    shape_to_strides(tiny_vector<V, N, R> const & shape,
                     tags::memory_order order = c_order)
    {
        tiny_vector<promote_type_t<V>, N> res(shape.size(), dont_init);

        if(order == c_order)
        {
            res[shape.size()-1] = 1;
            for(index_t k=shape.size()-2; k >= 0; --k)
                res[k] = res[k+1] * shape[k+1];
        }
        else
        {
            res[0] = 1;
            for(decltype(shape.size()) k=1; k < shape.size(); ++k)
                res[k] = res[k-1] * shape[k-1];
        }
        return res;
    }

    /****************************/
    /* tiny_vector manipulation */
    /****************************/

        /// reversed copy
    template <class V, index_t N, class R>
    inline
    tiny_vector<V, N>
    reversed(tiny_vector<V, N, R> const & v)
    {
        tiny_vector<V, N> res(v.size(), dont_init);
        for(decltype(v.size()) k=0; k<v.size(); ++k)
            res[k] = v[v.size()-1-k];
        return res;
    }

        /** \brief transposed copy

            Elements are arranged such that <tt>res[k] = v[permutation[k]]</tt>.
        */
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline
    tiny_vector<V1, N1>
    transposed(tiny_vector<V1, N1, R1> const & v,
               tiny_vector<V2, N2, R2> const & permutation)
    {
        vigra_precondition(v.size() == permutation.size(),
            "transposed(tiny_array, permutation): size mismatch.");
        tiny_vector<V1, N1> res(v.size(), dont_init);
        for(decltype(v.size()) k=0; k < v.size(); ++k)
        {
            XVIGRA_ASSERT_MSG(permutation[k] >= 0 && permutation[k] < (V2)v.size(),
                "transposed(tiny_array, permutation):  permutation index out of bounds.");
            res[k] = v[permutation[k]];
        }
        return res;
    }

    template <class V, index_t N, class R>
    inline
    tiny_vector<V, N>
    transposed(tiny_vector<V, N, R> const & v)
    {
        return reversed(v);
    }

    /**************************/
    /* tiny_vector arithmetic */
    /**************************/

    namespace tiny_detail
    {
        template <index_t N1, index_t N2>
        struct size_promote
        {
            static const index_t value  = N1;
            static const bool valid = (N1 == N2);
        };

        template <index_t N1>
        struct size_promote<N1, runtime_size>
        {
            static const index_t value  = runtime_size;
            static const bool valid = true;
        };

        template <index_t N2>
        struct size_promote<runtime_size, N2>
        {
            static const index_t value  = runtime_size;
            static const bool valid = true;
        };

        template <>
        struct size_promote<runtime_size, runtime_size>
        {
            static const index_t value  = runtime_size;
            static const bool valid = true;
        };
    }

    #define XVIGRA_TINYARRAY_OPERATORS(OP)                                                   \
    template <class V1, index_t N1, class R1, class V2,                                      \
              VIGRA_REQUIRE<!tiny_vector_concept<V2>::value &&                               \
                            std::is_convertible<V2, V1>::value> >                            \
    inline tiny_vector<V1, N1, R1> &                                                         \
    operator OP##=(tiny_vector<V1, N1, R1> & l,                                              \
                   V2 r)                                                                     \
    {                                                                                        \
        for(decltype(l.size()) i=0; i<l.size(); ++i)                                         \
            l[i] OP##= r;                                                                    \
        return l;                                                                            \
    }                                                                                        \
                                                                                             \
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>                \
    inline tiny_vector<V1, N1, R1> &                                                         \
    operator OP##=(tiny_vector<V1, N1, R1> & l,                                              \
                   tiny_vector<V2, N2, R2> const & r)                                        \
    {                                                                                        \
        XVIGRA_ASSERT_MSG(l.size() == r.size(),                                              \
            "tiny_vector::operator" #OP "=(): size mismatch.");                              \
        for(decltype(l.size()) i=0; i<l.size(); ++i)                                         \
            l[i] OP##= r[i];                                                                 \
        return l;                                                                            \
    }                                                                                        \
                                                                                             \
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>                \
    inline                                                                                   \
    tiny_vector<decltype((*(V1*)0) OP (*(V2*)0)), tiny_detail::size_promote<N1, N2>::value>  \
    operator OP(tiny_vector<V1, N1, R1> const & l,                                           \
                tiny_vector<V2, N2, R2> const & r)                                           \
    {                                                                                        \
        static_assert(tiny_detail::size_promote<N1, N2>::valid,                              \
            "tiny_vector::operator" #OP "(): size mismatch.");                               \
        XVIGRA_ASSERT_MSG(l.size() == r.size(),                                              \
            "tiny_vector::operator" #OP "(): size mismatch.");                               \
        tiny_vector<decltype((*(V1*)0) OP (*(V2*)0)),                                        \
                   tiny_detail::size_promote<N1, N2>::value> res(l.size(), dont_init);       \
        for(decltype(l.size()) i=0; i<l.size(); ++i)                                         \
            res[i] = l[i] OP r[i];                                                           \
        return res;                                                                          \
    }                                                                                        \
                                                                                             \
    template <class V1, index_t N1, class R1, class V2,                                      \
              VIGRA_REQUIRE<!tiny_vector_concept<V2>::value &&                               \
                            std::is_convertible<V2, V1>::value>>                             \
    inline                                                                                   \
    tiny_vector<decltype((*(V1*)0) OP (*(V2*)0)), N1>                                        \
    operator OP(tiny_vector<V1, N1, R1> const & l,                                           \
                V2 r)                                                                        \
    {                                                                                        \
        tiny_vector<decltype((*(V1*)0) OP (*(V2*)0)), N1> res(l.size(), dont_init);          \
        for(decltype(l.size()) i=0; i<l.size(); ++i)                                         \
            res[i] = l[i] OP r;                                                              \
        return res;                                                                          \
    }                                                                                        \
                                                                                             \
    template <class V1, class V2, index_t N2, class R2,                                      \
              VIGRA_REQUIRE<!tiny_vector_concept<V1>::value &&                               \
                            std::is_convertible<V1, V2>::value>>                             \
    inline                                                                                   \
    tiny_vector<decltype((*(V1*)0) OP (*(V2*)0)), N2>                                        \
    operator OP(V1 l,                                                                        \
                tiny_vector<V2, N2, R2> const & r)                                           \
    {                                                                                        \
        tiny_vector<decltype((*(V1*)0) OP (*(V2*)0)), N2> res(r.size(), dont_init);          \
        for(decltype(r.size()) i=0; i<r.size(); ++i)                                         \
            res[i] = l OP r[i];                                                              \
        return res;                                                                          \
    }

    XVIGRA_TINYARRAY_OPERATORS(+)
    XVIGRA_TINYARRAY_OPERATORS(-)
    XVIGRA_TINYARRAY_OPERATORS(*)
    XVIGRA_TINYARRAY_OPERATORS(/)
    XVIGRA_TINYARRAY_OPERATORS(%)
    XVIGRA_TINYARRAY_OPERATORS(&)
    XVIGRA_TINYARRAY_OPERATORS(|)
    XVIGRA_TINYARRAY_OPERATORS(^)
    XVIGRA_TINYARRAY_OPERATORS(<<)
    XVIGRA_TINYARRAY_OPERATORS(>>)

    #undef XVIGRA_TINYARRAY_OPERATORS

        /// Arithmetic identity
    template <class V, index_t N, class R>
    inline
    tiny_vector<V, N, R> const &
    operator+(tiny_vector<V, N, R> const & v)
    {
        return v;
    }

        /// Arithmetic negation
    template <class V, index_t N, class R>
    inline
    tiny_vector<decltype(-(*(V*)0)), N>
    operator-(tiny_vector<V, N, R> const & v)
    {
        tiny_vector<decltype(-(*(V*)0)), N> res(v.size(), dont_init);
        for(decltype(v.size()) k=0; k < v.size(); ++k)
            res[k] = -v[k];
        return res;
    }

        /// Boolean negation
    template <class V, index_t N, class R>
    inline
    tiny_vector<decltype(!(*(V*)0)), N>
    operator!(tiny_vector<V, N, R> const & v)
    {
        tiny_vector<decltype(!(*(V*)0)), N> res(v.size(), dont_init);
        for(decltype(v.size()) k=0; k < v.size(); ++k)
            res[k] = !v[k];
        return res;
    }

        /// Bitwise negation
    template <class V, index_t N, class R>
    inline
    tiny_vector<decltype(~(*(V*)0)), N>
    operator~(tiny_vector<V, N, R> const & v)
    {
        tiny_vector<decltype(~(*(V*)0)), N> res(v.size(), dont_init);
        for(decltype(v.size()) k=0; k < v.size(); ++k)
            res[k] = ~v[k];
        return res;
    }

    #define XVIGRA_TINYARRAY_UNARY_FUNCTION(FCT)                            \
    template <class V, index_t N, class R>                                  \
    inline auto                                                             \
    FCT(tiny_vector<V, N, R> const & v)                                     \
    {                                                                       \
        using math::FCT;                                                    \
        tiny_vector<decltype(FCT(v[0])), N> res(v.size(), dont_init);       \
        for(decltype(v.size()) k=0; k < v.size(); ++k)                      \
            res[k] = FCT(v[k]);                                             \
        return res;                                                         \
    }

    XVIGRA_TINYARRAY_UNARY_FUNCTION(abs)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(fabs)

    XVIGRA_TINYARRAY_UNARY_FUNCTION(cos)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(sin)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(tan)
    // XVIGRA_TINYARRAY_UNARY_FUNCTION(sin_pi)
    // XVIGRA_TINYARRAY_UNARY_FUNCTION(cos_pi)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(acos)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(asin)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(atan)

    XVIGRA_TINYARRAY_UNARY_FUNCTION(cosh)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(sinh)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(tanh)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(acosh)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(asinh)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(atanh)

    XVIGRA_TINYARRAY_UNARY_FUNCTION(sqrt)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(cbrt)
    // XVIGRA_TINYARRAY_UNARY_FUNCTION(sq)
    // XVIGRA_TINYARRAY_UNARY_FUNCTION(elementwise_norm)
    // XVIGRA_TINYARRAY_UNARY_FUNCTION(elementwise_squared_norm)

    XVIGRA_TINYARRAY_UNARY_FUNCTION(exp)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(exp2)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(expm1)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(log)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(log2)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(log10)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(log1p)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(logb)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(ilogb)

    XVIGRA_TINYARRAY_UNARY_FUNCTION(ceil)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(floor)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(trunc)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(round)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(lround)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(llround)
    // XVIGRA_TINYARRAY_UNARY_FUNCTION(even)
    // XVIGRA_TINYARRAY_UNARY_FUNCTION(odd)
    // XVIGRA_TINYARRAY_UNARY_FUNCTION(sign)
    // XVIGRA_TINYARRAY_UNARY_FUNCTION(signi)

    XVIGRA_TINYARRAY_UNARY_FUNCTION(erf)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(erfc)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(tgamma)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(lgamma)

    XVIGRA_TINYARRAY_UNARY_FUNCTION(conj)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(real)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(imag)
    XVIGRA_TINYARRAY_UNARY_FUNCTION(arg)

    #undef XVIGRA_TINYARRAY_UNARY_FUNCTION

    #define XVIGRA_TINYARRAY_BINARY_FUNCTION(FCT)                                               \
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>                   \
    inline auto                                                                                 \
    FCT(tiny_vector<V1, N1, R1> const & l,                                                      \
        tiny_vector<V2, N2, R2> const & r)                                                      \
    {                                                                                           \
        using math::FCT;                                                                        \
        static_assert(tiny_detail::size_promote<N1, N2>::valid,                                 \
            #FCT "(tiny_vector, tiny_vector): size mismatch.");                                 \
        XVIGRA_ASSERT_MSG(l.size() == r.size(),                                                 \
            #FCT "(tiny_vector, tiny_vector): size mismatch.");                                 \
        tiny_vector<decltype(FCT(l[0], r[0])),                                                  \
                   tiny_detail::size_promote<N1, N2>::value> res(l.size(), dont_init);          \
        for(decltype(l.size()) k=0; k < l.size(); ++k)                                          \
            res[k] = FCT(l[k], r[k]);                                                           \
        return res;                                                                             \
    }

    XVIGRA_TINYARRAY_BINARY_FUNCTION(atan2)
    XVIGRA_TINYARRAY_BINARY_FUNCTION(copysign)
    XVIGRA_TINYARRAY_BINARY_FUNCTION(fdim)
    XVIGRA_TINYARRAY_BINARY_FUNCTION(fmax)
    XVIGRA_TINYARRAY_BINARY_FUNCTION(fmin)
    XVIGRA_TINYARRAY_BINARY_FUNCTION(fmod)
    XVIGRA_TINYARRAY_BINARY_FUNCTION(hypot)

    #undef XVIGRA_TINYARRAY_BINARY_FUNCTION

        /** Apply pow() function to each vector element.
        */
    template <class V, index_t N, class R, class E>
    inline auto
    pow(tiny_vector<V, N, R> const & v, E exponent)
    {
        using math::pow;
        auto e = static_cast<promote_type_t<V, E>>(exponent);
        tiny_vector<decltype(pow(v[0], e)), N> res(v.size(), dont_init);
        for(decltype(v.size()) k=0; k < v.size(); ++k)
            res[k] = pow(v[k], e);
        return res;
    }

        /// sum of the vector's elements
    template <class V, index_t N, class R>
    inline auto
    sum(tiny_vector<V, N, R> const & v)
    {
        using result_type = decltype(v[0] + v[0]);
        result_type res = result_type();
        for(decltype(v.size()) k=0; k < v.size(); ++k)
            res += v[k];
        return res;
    }

        /// mean of the vector's elements
    template <class V, index_t N, class R>
    inline auto
    mean(tiny_vector<V, N, R> const & v)
    {
        using result_type = real_promote_type_t<decltype(sum(v))>;
        const result_type sumVal = static_cast<result_type>(sum(v));
        if(v.size() > 0)
            return sumVal / static_cast<result_type>(v.size());
        else
            return sumVal;
    }

        /// cumulative sum of the vector's elements
    template <class V, index_t N, class R>
    inline auto
    cumsum(tiny_vector<V, N, R> const & v)
    {
        using promote_type = decltype(v[0] + v[0]);
        tiny_vector<promote_type, N> res(v);
        for(decltype(v.size()) k=1; k < v.size(); ++k)
            res[k] += res[k-1];
        return res;
    }

        /// product of the vector's elements
    template <class V, index_t N, class R>
    inline auto
    prod(tiny_vector<V, N, R> const & v)
    {
        using result_type = decltype(v[0] * v[0]);
        if(v.size() == 0)
            return result_type();
        result_type res = v[0];
        for(decltype(v.size()) k=1; k < v.size(); ++k)
            res *= v[k];
        return res;
    }

        /// cumulative product of the vector's elements
    template <class V, index_t N, class R>
    inline auto
    cumprod(tiny_vector<V, N, R> const & v)
    {
        using promote_type = decltype(v[0] * v[0]);
        tiny_vector<promote_type, N> res(v);
        for(decltype(v.size()) k=1; k < v.size(); ++k)
            res[k] *= res[k-1];
        return res;
    }

        /// element-wise minimum
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline auto
    min(tiny_vector<V1, N1, R1> const & l,
        tiny_vector<V2, N2, R2> const & r)
    {
        using std::min;
        using promote_type = promote_type_t<V1, V2>;
        static_assert(tiny_detail::size_promote<N1, N2>::valid,
            "min(tiny_vector, tiny_vector): size mismatch.");
        XVIGRA_ASSERT_MSG(l.size() == r.size(),
            "min(tiny_vector, tiny_vector): size mismatch.");
        tiny_vector<promote_type, tiny_detail::size_promote<N1, N2>::value> res(l.size(), dont_init);
        for(decltype(l.size()) k=0; k < l.size(); ++k)
            res[k] = min(static_cast<promote_type>(l[k]), static_cast<promote_type>(r[k]));
        return res;
    }

        /// element-wise minimum with a constant
    template <class V1, index_t N1, class R1, class V2,
              VIGRA_REQUIRE<!tiny_vector_concept<V2>::value>>
    inline auto
    min(tiny_vector<V1, N1, R1> const & l,
        V2 const & r)
    {
        using std::min;
        using promote_type = promote_type_t<V1, V2>;
        tiny_vector<promote_type, N1> res(l.size(), dont_init);
        for(decltype(l.size()) k=0; k < l.size(); ++k)
            res[k] =  min(static_cast<promote_type>(l[k]), static_cast<promote_type>(r));
        return res;
    }

        /// element-wise minimum with a constant
    template <class V1, class V2, index_t N2, class R2,
              VIGRA_REQUIRE<!tiny_vector_concept<V1>::value>>
    inline auto
    min(V1 const & l, tiny_vector<V2, N2, R2> const & r)
    {
        return min(r, l);
    }

        /// minimal element
    template <class V, index_t N, class R>
    inline V const &
    min(tiny_vector<V, N, R> const & l)
    {
        index_t m = min_element(l);
        vigra_precondition(m >= 0, "min() of an empty tiny_vector is undefined.");
        return l[m];
    }

        /** Index of minimal element.

            Returns -1 for an empty array.
        */
    template <class V, index_t N, class R>
    inline index_t
    min_element(tiny_vector<V, N, R> const & l)
    {
        if(l.size() == 0)
            return -1;
        index_t m = 0;
        for(decltype(l.size()) i=1; i<l.size(); ++i)
            if(l[i] < l[m])
                m = i;
        return m;
    }

        /// element-wise maximum
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline auto
    max(tiny_vector<V1, N1, R1> const & l,
        tiny_vector<V2, N2, R2> const & r)
    {
        using std::max;
        using promote_type = promote_type_t<V1, V2>;
        static_assert(tiny_detail::size_promote<N1, N2>::valid,
            "max(tiny_vector, tiny_vector): size mismatch.");
        XVIGRA_ASSERT_MSG(l.size() == r.size(),
            "max(tiny_vector, tiny_vector): size mismatch.");
        tiny_vector<promote_type, tiny_detail::size_promote<N1, N2>::value> res(l.size(), dont_init);
        for(decltype(l.size()) k=0; k < l.size(); ++k)
            res[k] = max(static_cast<promote_type>(l[k]), static_cast<promote_type>(r[k]));
        return res;
    }

        /// element-wise maximum with a constant
    template <class V1, index_t N1, class R1, class V2,
              VIGRA_REQUIRE<!tiny_vector_concept<V2>::value>>
    inline auto
    max(tiny_vector<V1, N1, R1> const & l,
        V2 const & r)
    {
        using std::max;
        using promote_type = promote_type_t<V1, V2>;
        tiny_vector<promote_type, N1> res(l.size(), dont_init);
        for(decltype(l.size()) k=0; k < l.size(); ++k)
            res[k] =  max(static_cast<promote_type>(l[k]), static_cast<promote_type>(r));
        return res;
    }

        /// element-wise maximum with a constant
    template <class V1, class V2, index_t N2, class R2,
              VIGRA_REQUIRE<!tiny_vector_concept<V1>::value>>
    inline auto
    max(V1 const & l, tiny_vector<V2, N2, R2> const & r)
    {
        return max(r, l);
    }

        /// maximal element
    template <class V, index_t N, class R>
    inline V const &
    max(tiny_vector<V, N, R> const & l)
    {
        index_t m = max_element(l);
        vigra_precondition(m >= 0, "max() of an empty tiny_vector is undefined.");
        return l[m];
    }

        /** Index of maximal element.

            Returns -1 for an empty array.
        */
    template <class V, index_t N, class R>
    inline index_t
    max_element(tiny_vector<V, N, R> const & l)
    {
        if(l.size() == 0)
            return -1;
        index_t m = 0;
        for(decltype(l.size()) i=1; i<l.size(); ++i)
            if(l[i] > l[m])
                m = i;
        return m;
    }

        /** \brief Clip values below a threshold.

            All elements smaller than \a val are set to \a val.
        */
    template <class V, index_t N, class R>
    inline auto
    clip_lower(tiny_vector<V, N, R> const & t, const V val)
    {
        tiny_vector<V, N> res(t.size(), dont_init);
        for(decltype(t.size()) k=0; k < t.size(); ++k)
        {
            res[k] = t[k] < val ? val :  t[k];
        }
        return res;
    }

        /** \brief Clip values above a threshold.

            All elements bigger than \a val are set to \a val.
        */
    template <class V, index_t N, class R>
    inline auto
    clip_upper(tiny_vector<V, N, R> const & t, const V val)
    {
        tiny_vector<V, N> res(t.size(), dont_init);
        for(decltype(t.size()) k=0; k < t.size(); ++k)
        {
            res[k] = t[k] > val ? val :  t[k];
        }
        return res;
    }

        /** \brief Clip values to an interval.

            All elements less than \a valLower are set to \a valLower, all elements
            bigger than \a valUpper are set to \a valUpper.
        */
    template <class V, index_t N, class R>
    inline auto
    clip(tiny_vector<V, N, R> const & t,
         const V valLower, const V valUpper)
    {
        tiny_vector<V, N> res(t.size(), dont_init);
        for(decltype(t.size()) k=0; k < t.size(); ++k)
        {
            res[k] =  (t[k] < valLower)
                           ? valLower
                           : (t[k] > valUpper)
                                 ? valUpper
                                 : t[k];
        }
        return res;
    }

        /** \brief Clip values to a vector of intervals.

            All elements less than \a valLower are set to \a valLower, all elements
            bigger than \a valUpper are set to \a valUpper.
        */
    template <class V, index_t N1, class R1, index_t N2, class R2, index_t N3, class R3>
    inline auto
    clip(tiny_vector<V, N1, R1> const & t,
         tiny_vector<V, N2, R2> const & valLower,
         tiny_vector<V, N3, R3> const & valUpper)
    {
        XVIGRA_ASSERT_MSG(t.size() == valLower.size() && t.size() == valUpper.size(),
            "clip(): size mismatch.");
        tiny_vector<V, N1> res(t.size(), dont_init);
        for(decltype(t.size()) k=0; k < t.size(); ++k)
        {
            res[k] =  (t[k] < valLower[k])
                           ? valLower[k]
                           : (t[k] > valUpper[k])
                                 ? valUpper[k]
                                 : t[k];
        }
        return res;
    }

        /// dot product of two tiny_arrays
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline auto
    dot(tiny_vector<V1, N1, R1> const & l,
        tiny_vector<V2, N2, R2> const & r)
    {
        XVIGRA_ASSERT_MSG(l.size() == r.size(),
            "dot(tiny_vector, tiny_vector): size mismatch.");
        using result_type = decltype(l[0] * r[0]);
        result_type res = result_type();
        for(decltype(l.size()) k=0; k < l.size(); ++k)
            res += l[k] * r[k];
        return res;
    }

        /// cross product of two tiny_arrays
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline auto
    cross(tiny_vector<V1, N1, R1> const & r1,
          tiny_vector<V2, N2, R2> const & r2)
    {
        XVIGRA_ASSERT_MSG(r1.size() == 3 && r2.size() == 3,
            "cross(tiny_vector, tiny_vector): cross product requires size() == 3.");
        using result_type = tiny_vector<decltype(r1[0] * r2[0]), 3>;
        return  result_type{r1[1]*r2[2] - r1[2]*r2[1],
                            r1[2]*r2[0] - r1[0]*r2[2],
                            r1[0]*r2[1] - r1[1]*r2[0]};
    }

    #define XVIGRA_EMPTY
    #define XVIGRA_COMMA ,
    #define XVIGRA_ARGUMENT tiny_vector<V, N, R>
    #define XVIGRA_NORM_FUNCTION(NAME, RESULT_TYPE, REDUCE_EXPR, REDUCE_OP)     \
    template <class V, index_t N, class R>                                      \
    inline auto NAME(tiny_vector<V, N, R> const & t) noexcept                   \
    {                                                                           \
        using result_type = RESULT_TYPE;                                        \
        result_type result = result_type();                                     \
        for(decltype(t.size()) i=0; i<t.size(); ++i)                            \
            result = REDUCE_EXPR(result REDUCE_OP NAME(t[i]));                  \
        return result;                                                          \
    }

    XVIGRA_NORM_FUNCTION(norm_l0, unsigned long long, XVIGRA_EMPTY, +)
    XVIGRA_NORM_FUNCTION(norm_l1, squared_norm_type_t<XVIGRA_ARGUMENT>, XVIGRA_EMPTY, +)
    XVIGRA_NORM_FUNCTION(norm_sq, squared_norm_type_t<XVIGRA_ARGUMENT>, XVIGRA_EMPTY, +)
    XVIGRA_NORM_FUNCTION(norm_linf, decltype(norm_linf(std::declval<V>())),
                                    max, XVIGRA_COMMA)

    #undef XVIGRA_EMPTY
    #undef XVIGRA_COMMA
    #undef XVIGRA_ARGUMENT
    #undef XVIGRA_NORM_FUNCTION

    template <class V, index_t N, class R>
    inline auto norm_lp_to_p(tiny_vector<V, N, R> const & t, double p) noexcept
    {
        using result_type = norm_type_t<typename tiny_vector<V, N, R>::value_type>;
        result_type result = result_type();
        for(decltype(t.size()) i=0; i<t.size(); ++i)
            result = result + norm_lp_to_p(t[i], p);
        return result;
    }

    template <class V, index_t N, class R>
    inline auto norm_lp(tiny_vector<V, N, R> const & t, double p) noexcept
    {
        return std::pow(norm_lp_to_p(t, p), 1.0 / p);
    }
//@}

} // namespace xvigra

#endif // XVIGRA_TINY_VECTOR_HPP
