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
    
    /******************/
    /* tensor_concept */
    /******************/

    template <class T>
    struct tensor_concept
    : public xt::is_xexpression<T>
    {};
    
    /***********************/
    /* tiny_vector_concept */
    /***********************/

   template <class T>
    struct tiny_vector_concept
    : public std::integral_constant<bool,
                                    std::is_base_of<tags::tiny_vector_tag, std::decay_t<T>>::value>
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

            using type = decltype(test((std::iterator_traits<T> *)0));
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
