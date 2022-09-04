
# Julia Essentials

using BenchmarkTools
# Primitive data types
x = true
typeof(x)
y = 1>2

typeof(1.0)
typeof(1)

x = 2; y = 1.0
2x - 3y
@show 2x-3y
@show x +y;

# Complex numbers
x = 1+2im
y = 1-2im
x*y

# String
x = 10; y = 20
"x = $x"
"x+y = $(x+y)"

# concatenation
"foo"*"bar"
s = "Charlie don't surf"
split(s)
replace(s, "surf" => "ski")
split("fee, fi, fo", ",")
x = "foo", 1

# Containers
function g()
    return "foo", 1
end

g()

word, val = x
println("word = $word, val = $val")

# Dictionaries
d = Dict("name" => "Frodo", "age" => 33)
d["name"]
d["age"]

# Iterating
actions = ["surf", "ski"]
for action in actions
    println("Charlie doesn't $action")
end

keys(d) # iterator
collect(keys(d))

x_values = 1:5
for x in x_values
    println(x*x)
end

countries = ("Japan", "Korea", "China")
cities = ("Tokyo", "Seoul", "Beijing")
for (country, city) in zip(countries, cities)
    println("the capital of $country is $city")
end

# Comprehensions
[i+j for i in 1:3, j in 4:6]

[i + j + k for i in 1:3, j in 4:6, k in 7:9]

animals = ["dog", "cat", "bird"]

[(i, j) for i in 1:2, j in animals]

# Generators
xs = 1:10000
f(x) = x^2
f_x = f.(xs)
@btime sum([f(x) for x in $xs])
@btime sum($f_x)
@btime sum(f(x) for x in $xs)

# Comparisons
x = 1
x == 2
x != 3
1 + 1e-8 ≈ 1

true && false


# User-defined functions
# improve clarity of code by
    # separating different strands of logic
    # facilitating code reuse

map(x -> sin(1/x), randn(3))
f(x, a=1) = exp(cos(a*x))
f(π)

# Broadcasting

# Can be done via loop
x_vec = [2.0, 4.0, 6.0, 8.0]
y_vec = similar(x_vec)

for (i, x) in enumerate(x_vec)
    y_vec[i] = sin(x)
end

y_vec = sin.(x_vec)

# Simpler via Broadcasting
y_vec = sin.(x_vec)

function chisq(k)
    @assert k >0
    z = randn(k)
    return sum(z ->z^2, z)
end

chisq.([2, 4, 6])

x = 1.0:1.0:5.0
y = [ 2.0, 4.0, 5.0, 6.0, 8.0]
z = similar(y)
@. z = x + y - sin(x)

f(x, y) = dot([1, 2, 3],x) + y
f([3, 4, 5], 2)

x = 1.0:1.0:5.0
y = [2.0, 4.0, 5.0, 6.0, 8.0]
z = similar(y)

# broadcasting syntax; fuses operations
z .= x .+ y .- sin.(x)

@. z = x + y - sin(x)

f(a, b) = a + b
a = [1 2 3]
b = [4 5 6]
@show f.(a, b)
@show f.(a, 2)

# Scopes and closures
f(x) = x^2
# x is not bound to anything in this outer scope
y = 5
f(y)

function g()
    f(x) = x^2
    # x is not bound to anything in this outer scope
    y = 5
    f(y)
end
g()

# Local scope also works with named tuples
x = 0.1
y = 2
(x=x,  y=y)

# Similar with broadcasting
f(x) = x^2
x = 1:5
f.(x)

# Closures: want a function that calculates a value given some fixed parameters
f(x, a) = a*x^2
f(1, 0.2)

# Closure: capture variable from outer scope
function g(a)
    f(x) = a*x^2
    f(1)
end
g(0.2)

function solvemodel(x)
    a = x^2
    b = 2*a
    c = a + b
    return (a=a, b=b, c=c)
end

solvemodel(0.1)

# Higher-order functions
twice(f, x) = f(f(x))
f(x) = x^2
@show twice(f, 2.0)

twice(x -> x^2, 2.0)
a = 5
g(x) = a*x
@show twice(g, 2.0)

twice(f, 2.0)
twice(g, 2.0)

# Expectation of a function of a random variable following a specific distribution
using Expectations, Distributions
@show d = Exponential(2.0)
f(x) = x^2
@show expectation(f, d)


function multiplyit(a, g)
    return x -> a*g(x)
end

f(x) = x^2
h = multiplyit(2.0, f)
h(2)

# Exercises
# 1)
x_vals = 1:500
y_vals = 1:500
# Sum product of pairs of x_vals, y_vals

@btime sum(x*y for (x, y) in zip(x_vals,  y_vals))

@btime sum(x_vals.*y_vals)

# 2)
# Evaluate polynomial at a particular point: a_0 + a_1x + a_2x^2 + ...
function poly(x, coeff)
    z = 0.0
    for (i, coeff) in enumerate(coeff)
        z += coeff*x^(i-1)
        # shorthand for z = z + ...
    end
    return z
end

poly(1, (2, 4))
# More compactly

poly_2(x, coeff) = sum(a*x^(i-1) for (i,a) in enumerate(coeff))
poly_2(1, (2, 4))

# 3) Capital letters
# count number of capital letters in string
function f(string::String)
    count = 0
    for letter in string
        if (letter == uppercase(letter)) && isletter(letter)
            count += 1
        end
    end
    return count
end

f("The Rain in Spain")

# 4) Return true if every element of sequence a is in b

# check if every element of sequence a is also in sequence b
function f_seq(a, b)
    count = 0
    for (i, a_i) in enumerate(a)
        if a_i ∈ b
            count += 1
        end
    end
    return count == length(a)
end

println(f_seq([1, 2], [1, 2, 3]))
println(f_seq([1, 2, 3], [1, 2]))
# Use Set data type 
f_seq(seq_a, seq_b) = Set(seq_a) ⊆ Set(seq_b)


# 5)
# Calculate total population across cities
f_ex = open("us_cities.txt", "w") do f
    write(f,
  "new york: 8244910
  los angeles: 3819702
  chicago: 2707120
  houston: 2145146
  philadelphia: 1536471
  phoenix: 1469471
  san antonio: 1359758
  san diego: 1326179
  dallas: 1223229")
  end
f_ex = open("us_cities.txt", "r")
total_pop = 0
for line in eachline(f_ex)
    city, population = split(line, ':') #tuple unpacking
    total_pop += parse(Int, population)
end
close(f_ex)
println("Total population = $total_pop")





