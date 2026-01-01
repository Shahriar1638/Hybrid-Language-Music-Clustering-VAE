# CSE420 Assignment 3 - Question 1 Solution
## Grammar and Semantic Rules for Intermediate Code Generation

---

## Part A: Context-Free Grammar

### Production Rules:

```
1.  Program → FunctionList MainStmt
2.  FunctionList → Function FunctionList | Function

3.  Function → Type id ( Params ) { StmtList }
4.  Params → Type id | ε

5.  Type → int

6.  StmtList → Stmt StmtList | Stmt
7.  Stmt → Assignment | WhileStmt | ReturnStmt

8.  Assignment → id = Expr ;

9.  WhileStmt → while ( BoolExpr ) { StmtList }

10. ReturnStmt → return Expr ;

11. BoolExpr → Expr RelOp Expr
12. RelOp → <= | > | < | >= | == | !=

13. Expr → Expr + Term 
14. Expr → Expr - Term 
15. Expr → Term

16. Term → Term * Factor 
17. Term → Term / Factor 
18. Term → Factor

19. Factor → id 
20. Factor → num 
21. Factor → Call

22. Call → id ( Args )
23. Args → Expr | ε

24. MainStmt → Assignment
```

---

## Part B: Semantic Rules for Intermediate Code Generation

### Notation:
- **E.place** = Location (variable/temporary) holding the value of E
- **E.code** = Three-address code sequence for evaluating E
- **newtemp()** = Creates new temporary variable (t1, t2, t3, ...)
- **newlabel()** = Creates new label (L1, L2, L3, ...)
- **gen(...)** = Generates three-address instruction
- **||** = Concatenates code sequences

---

### 1. Program Structure

```
Production: Program → FunctionList MainStmt

Semantic Rule:
    Program.code = FunctionList.code || MainStmt.code
```

---

### 2. Function Definition

```
Production: Function → Type id ( Params ) { StmtList }

Semantic Rule:
    Function.code = gen('func begin' id.name) ||
                    Params.code ||
                    StmtList.code ||
                    gen('func end' id.name)
```

---

### 3. Assignment Statement

```
Production: Assignment → id = Expr ;

Semantic Rule:
    Assignment.code = Expr.code || 
                      gen(id.place '=' Expr.place)
    Assignment.place = id.place
```

**Example:**
For `res = 1;`
```
Code: res = 1
```

---

### 4. Arithmetic Expressions

#### Addition:
```
Production: Expr → Expr₁ + Term

Semantic Rule:
    Expr.place = newtemp()
    Expr.code = Expr₁.code || 
                Term.code || 
                gen(Expr.place '=' Expr₁.place '+' Term.place)
```

**Example:**
For `res * i`
```
t1 = res * i
```

#### Subtraction:
```
Production: Expr → Expr₁ - Term

Semantic Rule:
    Expr.place = newtemp()
    Expr.code = Expr₁.code || 
                Term.code || 
                gen(Expr.place '=' Expr₁.place '-' Term.place)
```

#### Multiplication:
```
Production: Term → Term₁ * Factor

Semantic Rule:
    Term.place = newtemp()
    Term.code = Term₁.code || 
                Factor.code || 
                gen(Term.place '=' Term₁.place '*' Factor.place)
```

---

### 5. While Loop

```
Production: WhileStmt → while ( BoolExpr ) { StmtList }

Semantic Rule:
    begin = newlabel()
    after = newlabel()
    
    WhileStmt.code = gen(begin ':') ||
                     BoolExpr.code ||
                     gen('ifFalse' BoolExpr.place 'goto' after) ||
                     StmtList.code ||
                     gen('goto' begin) ||
                     gen(after ':')
```

**Explanation:**
- `begin`: Label at start of loop (for jumping back)
- `after`: Label after loop (for exiting)
- If condition is false, jump to `after`
- Otherwise, execute statements and jump back to `begin`

---

### 6. Boolean Expression

```
Production: BoolExpr → Expr₁ RelOp Expr₂

Semantic Rule:
    BoolExpr.place = newtemp()
    BoolExpr.code = Expr₁.code || 
                    Expr₂.code || 
                    gen(BoolExpr.place '=' Expr₁.place RelOp.op Expr₂.place)
```

**Example:**
For `i <= x`
```
t1 = i <= x
```

---

### 7. Function Call

```
Production: Call → id ( Args )

Semantic Rule:
    Call.place = newtemp()
    Call.code = Args.code ||
                gen('param' Args.place) ||
                gen(Call.place '=' 'call' id.name ',' 1)
```

**Explanation** (from textbook reference):
- First evaluate the argument expression
- Generate `param` instruction to pass parameter
- Generate `call` instruction with function name and parameter count
- Store result in temporary

**Example:**
For `factorial(y)`
```
param y
t2 = call factorial, 1
```

---

### 8. Return Statement

```
Production: ReturnStmt → return Expr ;

Semantic Rule:
    ReturnStmt.code = Expr.code ||
                      gen('return' Expr.place)
```

**Example:**
For `return res;`
```
return res
```

---

### 9. Factor Productions

```
Production: Factor → id
Semantic Rule:
    Factor.place = id.place
    Factor.code = ''

Production: Factor → num
Semantic Rule:
    Factor.place = num.value
    Factor.code = ''

Production: Factor → Call
Semantic Rule:
    Factor.place = Call.place
    Factor.code = Call.code
```

---

## Summary of Three-Address Code Instructions Generated:

1. **Assignment**: `x = y`
2. **Binary Operation**: `x = y op z` where op ∈ {+, -, *, /}
3. **Relational Operation**: `x = y relop z` where relop ∈ {<, >, <=, >=, ==, !=}
4. **Unconditional Jump**: `goto L`
5. **Conditional Jump**: `ifFalse x goto L`
6. **Parameter Passing**: `param x`
7. **Function Call**: `x = call f, n` (f = function name, n = # of parameters)
8. **Return**: `return x`
9. **Function Begin/End**: `func begin f` and `func end f`
10. **Labels**: `L:`

---

## Key Points from Reference Material:

1. **Function calls are unraveled** into:
   - Parameter evaluation
   - `param` instructions
   - `call` instruction

2. **Attributes used**:
   - `.place`: holds the address/location
   - `.code`: holds generated instruction sequence

3. **Helper functions**:
   - `newtemp()`: generates fresh temporaries
   - `newlabel()`: generates fresh labels
   - `gen()`: creates three-address instructions

This grammar and semantic rules system follows the approach described in Chapter 6 of the compiler design textbook and will generate proper three-address intermediate code for the given factorial and fact_sum program.

---

# STEP-BY-STEP EXAMPLE: Applying Grammar and Semantic Rules

Let's take a **small portion** from the assignment code and show how to apply the grammar and semantic rules step by step.

## Example Code Fragment:
```c
int factorial(x) {
    res = 1;
    res = res * i;
    return res;
}
```

---

## Step 1: Parse the Function Declaration

### Derivation:
```
Program 
⇒ FunctionList MainStmt
⇒ Function MainStmt
⇒ Type id(Params) { StmtList } MainStmt
⇒ int factorial(Params) { StmtList } MainStmt
```

### Apply Semantic Rule for Function:
```
Function → int id(Params) { StmtList }

Semantic Action:
    Function.code = gen('func begin factorial') ||
                    StmtList.code ||
                    gen('func end factorial')
```

---

## Step 2: Parse First Assignment `res = 1;`

### Derivation:
```
StmtList
⇒ Stmt StmtList
⇒ Assignment StmtList
⇒ id = Expr ; StmtList
⇒ res = Expr ;
⇒ res = Term ;
⇒ res = Factor ;
⇒ res = num ;
⇒ res = 1 ;
```

### Apply Semantic Rules:

**For Factor → num:**
```
Factor.place = 1
Factor.code = ''
```

**For Term → Factor:**
```
Term.place = Factor.place = 1
Term.code = Factor.code = ''
```

**For Expr → Term:**
```
Expr.place = Term.place = 1
Expr.code = Term.code = ''
```

**For Assignment → id = Expr ;**
```
Assignment.code = Expr.code || gen('res = 1')
                = '' || gen('res = 1')
                = gen('res = 1')
```

### Generated Three-Address Code:
```
res = 1
```

---

## Step 3: Parse Second Assignment `res = res * i;`

### Derivation:
```
Assignment
⇒ id = Expr ;
⇒ res = Expr ;
⇒ res = Term ;
⇒ res = Term * Factor ;
⇒ res = Factor * Factor ;
⇒ res = id * id ;
⇒ res = res * i ;
```

### Apply Semantic Rules (Bottom-Up):

**For Factor → id (first occurrence "res"):**
```
Factor₁.place = 'res'
Factor₁.code = ''
```

**For Factor → id (second occurrence "i"):**
```
Factor₂.place = 'i'
Factor₂.code = ''
```

**For Term → Factor₁ (converts first factor to term):**
```
Term₁.place = Factor₁.place = 'res'
Term₁.code = Factor₁.code = ''
```

**For Term → Term₁ * Factor₂:**
```
Term.place = newtemp() = t1
Term.code = Term₁.code || Factor₂.code || gen('t1 = res * i')
          = '' || '' || gen('t1 = res * i')
          = gen('t1 = res * i')
```

**For Expr → Term:**
```
Expr.place = Term.place = t1
Expr.code = Term.code = gen('t1 = res * i')
```

**For Assignment → res = Expr ;**
```
Assignment.code = Expr.code || gen('res = t1')
                = gen('t1 = res * i') || gen('res = t1')
```

### Generated Three-Address Code:
```
t1 = res * i
res = t1
```

---

## Step 4: Parse Return Statement `return res;`

### Derivation:
```
ReturnStmt
⇒ return Expr ;
⇒ return Term ;
⇒ return Factor ;
⇒ return id ;
⇒ return res ;
```

### Apply Semantic Rules:

**For Factor → id:**
```
Factor.place = 'res'
Factor.code = ''
```

**For Term → Factor:**
```
Term.place = Factor.place = 'res'
Term.code = Factor.code = ''
```

**For Expr → Term:**
```
Expr.place = Term.place = 'res'
Expr.code = Term.code = ''
```

**For ReturnStmt → return Expr ;**
```
ReturnStmt.code = Expr.code || gen('return res')
                = '' || gen('return res')
                = gen('return res')
```

### Generated Three-Address Code:
```
return res
```

---

## Step 5: Complete Function Code

### Combine all statements:
```
StmtList.code = Assignment₁.code || Assignment₂.code || ReturnStmt.code
              = gen('res = 1') || 
                gen('t1 = res * i') || gen('res = t1') ||
                gen('return res')
```

### Apply Function semantic rule:
```
Function.code = gen('func begin factorial') ||
                StmtList.code ||
                gen('func end factorial')
```

---

## Final Generated Three-Address Code:
```
func begin factorial
    res = 1
    t1 = res * i
    res = t1
    return res
func end factorial
```

---

## Complete Example with While Loop

Let's show one more example with a **while loop**: `while(i <= x) { res = res * i; }`

### Derivation:
```
WhileStmt
⇒ while ( BoolExpr ) { StmtList }
⇒ while ( Expr RelOp Expr ) { StmtList }
⇒ while ( i <= x ) { Assignment }
⇒ while ( i <= x ) { res = res * i ; }
```

### Apply Semantic Rules:

**Step 1: Parse BoolExpr (i <= x)**

For left Expr (i):
```
Factor.place = 'i'
Term.place = 'i'
Expr₁.place = 'i'
Expr₁.code = ''
```

For right Expr (x):
```
Factor.place = 'x'
Term.place = 'x'
Expr₂.place = 'x'
Expr₂.code = ''
```

For BoolExpr:
```
BoolExpr.place = newtemp() = t2
BoolExpr.code = Expr₁.code || Expr₂.code || gen('t2 = i <= x')
              = '' || '' || gen('t2 = i <= x')
              = gen('t2 = i <= x')
```

**Step 2: Parse Statement inside loop (res = res * i)**

From previous example:
```
Assignment.code = gen('t3 = res * i') || gen('res = t3')
StmtList.code = Assignment.code
```

**Step 3: Apply While semantic rule**

```
begin = newlabel() = L1
after = newlabel() = L2

WhileStmt.code = gen('L1:') ||
                 BoolExpr.code ||
                 gen('ifFalse t2 goto L2') ||
                 StmtList.code ||
                 gen('goto L1') ||
                 gen('L2:')
```

### Generated Three-Address Code:
```
L1:
    t2 = i <= x
    ifFalse t2 goto L2
    t3 = res * i
    res = t3
    goto L1
L2:
```

---

## Key Takeaways:

1. **Bottom-up evaluation**: Start from terminals (id, num) and work up
2. **Synthesized attributes**: Each node's code depends on children's code
3. **Temporaries**: Generated for intermediate results (t1, t2, t3, ...)
4. **Labels**: Generated for control flow (L1, L2, ...)
5. **Code concatenation**: Use || to join code sequences

This shows exactly how the grammar rules and semantic actions work together to generate intermediate code!