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

#ifndef XVIGRA_CONCEPTS_HPP
#define XVIGRA_CONCEPTS_HPP

#include "global.hpp"
#include <xtensor/xconcepts.hpp>
#include <xtensor/xsemantic.hpp>
#include <xtensor/xstrided_view.hpp>
#include <type_traits>

namespace xvigra
{
    /*****************/
    /* VIGRA_REQUIRE */
    /*****************/

    #define VIGRA_REQUIRE class = std::enable_if_t

    /*********************/
    /* container_concept */
    /*********************/

    namespace detail
    {
        template <class T>
        struct container_concept_impl
        {
            static void test(...);

            template <class U>
            static int test(U *, typename U::value_type * = 0);

            static constexpr bool value = std::is_same<int, decltype(test((std::decay_t<T> *)0))>::value;
        };
    }

    template <class T>
    struct container_concept
    : public std::integral_constant<bool,
                                    detail::container_concept_impl<T>::value>
    {};

    /**********************/
    /* xt::is_xexpression */
    /**********************/

    using xt::is_xexpression;

    /******************/
    /* tensor_concept */
    /******************/

    namespace detail
    {
        template <class T>
        struct tensor_concept
        : public std::is_base_of<xt::xcontainer_semantic<std::decay_t<T>>, std::decay_t<T>>
        {};

        template <class T, index_t N>
        struct tensor_concept<view_nd<T, N>>
        : public std::true_type
        {};

        template <class T, index_t N, class A>
        struct tensor_concept<array_nd<T, N, A>>
        : public std::true_type
        {};

        template <class CT, class... S>
        struct tensor_concept<xt::xview<CT, S...>>
        : public std::true_type
        {};

        template <class T, class S, class D>
        struct tensor_concept<xt::xstrided_view<T, S, D>>
        : public std::true_type
        {};
    }

    template <class T>
    using tensor_concept = detail::tensor_concept<std::decay_t<T>>;

    /********************/
    /* has_raw_data_api */
    /********************/

    namespace detail
    {
        template <class T>
        struct has_raw_data_api
        : public std::is_base_of<xt::xcontainer_semantic<std::decay_t<T>>, std::decay_t<T>>
        {};

        template <class T, index_t N>
        struct has_raw_data_api<view_nd<T, N>>
        : public std::true_type
        {};

        template <class T, index_t N, class A>
        struct has_raw_data_api<array_nd<T, N, A>>
        : public std::true_type
        {};

        template <class T, class... S>
        struct has_raw_data_api<xt::xview<T, S...>>
        : public has_raw_data_api<typename xt::xview<T, S...>::xexpression_type>
        {};

        template <class T, class S, class D>
        struct has_raw_data_api<xt::xstrided_view<T, S, D>>
        : public has_raw_data_api<typename xt::xstrided_view<T, S, D>::xexpression_type>
        {};
    }

    template <class T>
    using has_raw_data_api = detail::has_raw_data_api<std::decay_t<T>>;

    /***********************/
    /* tiny_vector_concept */
    /***********************/

    template <class T>
    struct tiny_vector_concept
    : public std::false_type
    {};

    template <class V, index_t N, class R>
    struct tiny_vector_concept<tiny_vector<V, N, R>>
    : public std::true_type
    {};

    /*******************/
    /* view_nd_concept */
    /*******************/

    template <class T>
    struct view_nd_concept
    : public std::is_base_of<tags::view_nd_tag, std::decay_t<T>>
    {};

    /*********************/
    /* kernel_1d_concept */
    /*********************/

    template <class T>
    struct kernel_1d_concept
    : public std::is_base_of<tags::kernel_1d_tag, std::decay_t<T>>
    {};

    /*********************/
    /* iterator concepts */
    /*********************/

    namespace detail
    {
        template <class T>
        struct iterator_category_impl
        {
            static void test(...);

            template <class U>
            static typename U::iterator_category test(U *);

            using type = decltype(test((std::iterator_traits<std::decay_t<T>> *)0));
        };

        template <class T, class Category>
        struct iterator_concept_impl
        : public std::integral_constant<bool,
                                        std::is_convertible<typename iterator_category_impl<T>::type,
                                                            Category>::value>
        {};
    }

    template <class T>
    using input_iterator_concept = detail::iterator_concept_impl<T, std::input_iterator_tag>;

    template <class T>
    using output_iterator_concept = detail::iterator_concept_impl<T, std::output_iterator_tag>;

    template <class T>
    using forward_iterator_concept = detail::iterator_concept_impl<T, std::forward_iterator_tag>;

    template <class T>
    using bidirectional_iterator_concept = detail::iterator_concept_impl<T, std::bidirectional_iterator_tag>;

    template <class T>
    using random_access_iterator_concept = detail::iterator_concept_impl<T, std::random_access_iterator_tag>;

} // namespace xvigra

#endif // XVIGRA_CONCEPTS_HPP
