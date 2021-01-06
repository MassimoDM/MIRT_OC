# @Author: Massimo De Mauri <massimo>
# @Date:   2021-01-05T17:58:29+01:00
# @Email:  massimo.demauri@protonmail.com
# @Filename: CSfuncForJulia.py
# @Last modified by:   massimo
# @Last modified time: 2021-01-06T10:35:09+01:00
# @License: LGPL-3.0

import casadi as cs
import subprocess
import os

class CSfuncForJulia:
	def __init__(self,name,syms,expression):

		# info
		self.name = name
		self.main_sym = syms[0]
		self.param_syms = syms[1:]
		self.param_names = [s.name() for s in self.param_syms]
		self.main_sizes = (self.main_sym.numel(),expression.numel())
		self.param_sizes = tuple(self.param_syms[i].numel() for i in range(len(self.param_syms)))

		# expression evaluation
		self.eval = cs.Function('eval',syms,[expression])

		# expression jacobian
		jacobian = cs.jacobian(expression,syms[0])
		self.jac_nnz = jacobian.nnz()
		self.jac_sparsity = jacobian.sparsity().get_triplet()
		self.eval_jac = cs.Function('eval_jac',syms,[jacobian])

		# hessian of each of the elements of the expression
		hessian = [cs.hessian(expression[i],syms[0])[0] for i in range(expression.shape[0])]
		self.hes_nnz = [hessian[i].nnz() for i in range(expression.shape[0])]
		self.hes_sparsity = [hessian[i].sparsity().get_triplet() for i in range(expression.shape[0])]
		self.eval_hes = [cs.Function('eval_hes'+str(i),syms,[hessian[i]]) for i in range(expression.shape[0])]

		# location of the compiled library
		self.lib_path = None

	def compile(self,compilation_folder):
		# creates a clib (named: <self.name>.so) in <compilation_folder> with the above functions named: simplified_<func_name>
		# (the order of the inputs is the same of the one of the casadi functions defined above)

		funs = [self.eval,self.eval_jac]+self.eval_hes

		os.makedirs(compilation_folder,exist_ok=True)
		cg = cs.CodeGenerator(self.name)
		cg.add_include("stdlib.h")
		for f in funs:
			cg.add(f)
		cg.generate(compilation_folder+os.sep)

		with open(os.path.join(compilation_folder,self.name+".c"),"a") as out:
			out.write("""
						typedef struct {
						const double** arg;
						double** res;
						casadi_int* iw;
						double* w;
						void* mem;
						} casadi_mem;
						""")
			for f in funs:
				in_arg = ", ".join(["const double * in%d" % i for i in range(f.n_in())])
				out_arg = ", ".join(["double * out%d" % i for i in range(f.n_out())])
				set_in = "\n      ".join(["mem.arg[{i}] = in{i};".format(i=i) for i in range(f.n_in())])
				set_out = "\n      ".join(["mem.res[{i}] = out{i};".format(i=i) for i in range(f.n_out())])
				out.write("""
							CASADI_SYMBOL_EXPORT int simplified_{name}({in_arg}, {out_arg}) {{
							static int initialized = 0;
							static casadi_mem mem;
							if (!initialized) {{
							casadi_int sz_arg, sz_res, sz_iw, sz_w;
							{name}_work(&sz_arg, &sz_res, &sz_iw, &sz_w);
							mem.arg = malloc(sizeof(void*)*sz_arg);
							mem.res = malloc(sizeof(void*)*sz_res);
							mem.iw = malloc(sizeof(casadi_int)*sz_iw);
							mem.w = malloc(sizeof(casadi_int)*sz_w);
							{name}_incref();
							mem.mem = 0;//{name}_checkout();
							}};
							{set_in}
							{set_out}
							return {name}(mem.arg, mem.res, mem.iw, mem.w, mem.mem);
							}}
							""".format(name=f.name(), in_arg=in_arg, out_arg=out_arg, set_in=set_in, set_out=set_out))

		subprocess.Popen(["gcc","-O0","-shared","-fPIC",os.path.join(compilation_folder,self.name+".c"),"-o",os.path.join(compilation_folder,self.name+".so")]).wait()

		self.lib_path = compilation_folder+'/'+self.name+'.so'

	def pack_for_julia(self,param_values,compilation_folder=""):
		if self.lib_path is None:
			if len(compilation_folder)==0:
				raise NameError("pack_for_julia : CSfuncForClib not yet compiled")
			elif len(compilation_folder)>0:
				self.compile(compilation_folder)

		return {'param_values':[param_values[name] for name in self.param_names],
				'sizes':self.main_sizes,
				'jac_nnz':self.jac_nnz,
				'jac_sparsity':self.jac_sparsity,
				'hes_nnz':self.hes_nnz,
				'hes_sparsity':self.hes_sparsity,
				'lib_path':self.lib_path}


if __name__ == "__main__":
	from casadi import *
	DM.rng(0)
	A = DM(Sparsity.lower(5),DM.rand(15))
	x = MX.sym("x",5)
	y = MX.sym("y",5)

	z = A @ x+y
	c = CSfuncForClib('foo',[x,y],z)
	print("numerical check",c.eval_jac([1,2,3,4,5],[2,1,3,4,5]))
	c.compile("dest")
