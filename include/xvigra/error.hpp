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

#ifndef XVIGRA_ERROR_HPP
#define XVIGRA_ERROR_HPP

#include <stdexcept>

#ifdef XVIGRA_ENABLE_ASSERT
#define XVIGRA_TRY(expr)                                                                   \
    try                                                                                    \
    {                                                                                      \
        expr;                                                                              \
    }                                                                                      \
    catch (std::exception& e)                                                              \
    {                                                                                      \
        throw std::runtime_error(std::string(__FILE__) + ':' + std::to_string(__LINE__) +  \
              ": check raised exception\n\t" + std::string(e.what()) + "\n");              \
    }
#else
#define XVIGRA_TRY(expr)
#endif

#ifdef XVIGRA_ENABLE_ASSERT
#define XVIGRA_ASSERT(expr)                                                                \
    if (expr)                                                                              \
    {}                                                                                     \
    else                                                                                   \
    {                                                                                      \
        throw std::runtime_error(std::string(__FILE__) + ':' + std::to_string(__LINE__) +  \
            ": assertion failed (" #expr ").\n");                                          \
    }
#else
#define XVIGRA_ASSERT(expr)
#endif

#ifdef XVIGRA_ENABLE_ASSERT
#define XVIGRA_ASSERT_MSG(expr, msg)                                                       \
    if (expr)                                                                              \
    {}                                                                                     \
    else                                                                                   \
    {                                                                                      \
        throw std::runtime_error(std::string(__FILE__) + ':' + std::to_string(__LINE__) +  \
            ": " + msg + " (" #expr ").\n");                                               \
    }
#else
#define XVIGRA_ASSERT_MSG(expr, msg)
#endif

    /* put the exception in the 'else' branch to prevent incorrect 'else' association in code
       like this:
    
       if(condition) 
           vigra_precondition(other_condition, "failure");
       else   // belongs to the precondition when vigra_precondition didn't have an 'else'
           ...
    */
#define vigra_precondition(expr, msg)                                                      \
    if (expr)                                                                              \
    {}                                                                                     \
    else                                                                                   \
    {                                                                                      \
        throw std::runtime_error(std::string("Precondition violation!\n") + msg +          \
                     "\n  " + __FILE__ + '(' + std::to_string(__LINE__) + ")\n");          \
    }

#define vigra_fail(msg)  throw std::runtime_error(std::string(msg) +                       \
                         "\n  " + __FILE__ + '(' + std::to_string(__LINE__) + ")\n");

#endif  // XVIGRA_ERROR_HPP
