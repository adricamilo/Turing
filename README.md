# Turing

This is a Python simulator of a modified Turing machine. This construction was first suggested by Post[^1]. For a complete theorical description, see Chapter 1 of Davis[^2].

## Basic usage

To define a Turing machine, we use the `TuringMachine` class. For example:

```python
>>> from Turing import TuringMachine, Quadruple
>>> adder_machine = TuringMachine([
...     Quadruple(1, 1, '0', 1),
...     Quadruple(1, 0, 'R', 2),
...     Quadruple(2, 1, 'R', 2),
...     Quadruple(2, 0, 'R', 3),
...     Quadruple(3, 1, '0', 3),
... ])
```

This defines a Turing machine capable of adding nonnegative integers. We have:

```python
>>> adder_machine.compute([4,5])
[(q₁111110111111, None),
 (q₁011110111111, [q₁ 1 0 q₁]),
 (0q₂11110111111, [q₁ 0 R q₂]),
 (01q₂1110111111, [q₂ 1 R q₂]),
 (011q₂110111111, [q₂ 1 R q₂]),
 (0111q₂10111111, [q₂ 1 R q₂]),
 (01111q₂0111111, [q₂ 1 R q₂]),
 (011110q₃111111, [q₂ 0 R q₃]),
 (011110q₃011111, [q₃ 1 0 q₃])]
>>> adder_machine.resultant([4,5])
011110q₃011111
>>> adder_machine.resultant([4,5]).count_ones()
9
``` 

## Notes

This simulation does not (yet) support relative computations (see Chapter 1, Section 4, of Davis[^2]). Computations with quadruples of the form $q_i\;S_j\;q_k\;q_l$ will throw exceptions.

[^1] Emil L. Post. "Recursive Unsolvability of a Problem of Thue." In: *Journal of Symbolic
Logic* 12.1 (Mar. 1947), pp. 1–11. issn: 00224812. doi: `10.2307/2267170`. url:
`http://www.jstor.org/stable/2267170`.

[^2] Martin Davis. *Computability & Unsolvability*. Orig. publ.: New York, McGraw-Hill, 1958.
New York, NY: Dover Publications, Inc., 1982. 248 pp. isbn: 9780486614717.