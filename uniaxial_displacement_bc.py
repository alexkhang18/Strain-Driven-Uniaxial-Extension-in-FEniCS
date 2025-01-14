from dolfin import *

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {
    "optimize": True,
    "eliminate_zeros": True,
    "precompute_basis_const": True,
    "precompute_ip_const": True
}

# Define the corner points of the box for the mesh
point_1 = Point(0.0, 0.0, 0.0)  # Bottom-left-front corner
point_2 = Point(0.25, 0.5, 1.0)  # Top-right-back corner

# Define the number of divisions along each axis
nx, ny, nz = 5, 10, 20  # Mesh divisions in x, y, and z directions

# Create the box mesh
mesh = BoxMesh(point_1, point_2, nx, ny, nz)

# Define function spaces
V = VectorFunctionSpace(mesh, "Lagrange", 1)  # Vector function space
V2 = TensorFunctionSpace(mesh, "Lagrange", degree=1, shape=(3, 3))  # Tensor function space

# Define subdomains for parts of the boundaries
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)  # Left boundary at x = 0

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0)  # Right boundary at x = 1

class Front(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)  # Front boundary at y = 0

class Back(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0)  # Back boundary at y = 1

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], 0.0)  # Bottom boundary at z = 0

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2], 1.0)  # Top boundary at z = 1

# Initialize sub-domain instances
left = Left()
right = Right()
front = Front()
back = Back()
bottom = Bottom()
top = Top()

# Define a mesh function to mark boundary subdomains
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)  # Initialize all boundaries to 0
left.mark(boundaries, 1)
top.mark(boundaries, 2)
right.mark(boundaries, 3)
bottom.mark(boundaries, 4)
front.mark(boundaries, 5)
back.mark(boundaries, 6)

# Define the boundary measure
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# Initialize the displacement function for the top boundary
uz = Function(V)

# Define Dirichlet boundary conditions
c = Expression(("0.0", "0.0", "0.0"), degree=1)  # Zero displacement
bcbo = DirichletBC(V, c, bottom)  # Fixed bottom boundary
bct = DirichletBC(V, uz, top)  # Displacement applied on the top boundary
bcs = [bcbo, bct]

# Define functions for the variational problem
du = TrialFunction(V)  # Incremental displacement
v = TestFunction(V)  # Test function
u = Function(V)  # Displacement from the previous iteration
B = Constant((0.0, 0.0, 0.0))  # Body force per unit volume
T = Constant((0.0, 0.0, 0.0))  # Surface traction

# Define kinematic quantities
d = u.geometric_dimension()  # Space dimension
I = Identity(d)  # Identity tensor
F = I + grad(u)  # Deformation gradient
C = F.T * F  # Right Cauchy-Green tensor

# Invariants of the deformation tensor
Ic = tr(C)  # First invariant
J = det(F)  # Determinant of deformation gradient

# Elasticity parameters (compressible Neo-Hookean material)
E, nu = 10.0, 0.40  # Young's modulus and Poisson's ratio
mu = Constant(E / (2 * (1 + nu)))  # Shear modulus
lmbda = Constant(E * nu / ((1 + nu) * (1 - 2 * nu)))  # Lame's first parameter

# Stored strain energy density function
psi = (mu / 2) * (Ic - 3) - mu * ln(J) + (lmbda / 2) * (ln(J)) ** 2

# Total potential energy
Pi = psi * dx - dot(B, u) * dx - dot(T, u) * ds

# First variation of the potential energy
F = derivative(Pi, u, v)

# Jacobian of the first variation
J = derivative(F, u, du)

# Simulation parameters
chunks = 10  # Number of simulation steps
uz_max = 1.54  # Maximum displacement

# Create a file to store the displacements
file = File("displacement_u_driven.pvd")

# Perform simulations over the specified number of steps
for i in range(chunks + 1):
    print(f"Simulation step: {i}/{chunks}")

    # Assign current displacement for the top boundary
    uz.assign(Constant((0.0, 0.0, (i / chunks) * uz_max)))

    # Solve the variational problem
    solve(F == 0, u, bcs, J=J, form_compiler_parameters=ffc_options)

    # Save the displacement to the output file
    u.rename("Displacement", "label")
    file << (u, i / chunks)
