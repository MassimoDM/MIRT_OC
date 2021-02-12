# @Author: Massimo De Mauri <massimo>
# @Date:   2021-01-05T17:58:29+01:00
# @Email:  massimo.demauri@protonmail.com
# @Filename: CvxFuncForJulia.py
# @Last modified by:   massimo
# @Last modified time: 2021-01-14T18:44:44+01:00
# @License: LGPL-3.0

import casadi as cs
import MIRT_OC as oc
import subprocess
import os

class CvxFuncForJulia:
	def __init__(self,name,syms,expression):

		# info
		self.name = name
		self.sizes = (syms[0].numel(),expression.numel(),sum([s.numel() for s in syms[1:]]))

		# use a single parameter for the expression (because julia sucks sometimes)
		self.param_sym = oc.MX.sym("P",self.sizes[2])
		self.param_names = [s.name() for s in syms[1:]]
		expression_ = cs.substitute(expression,cs.vcat(syms[1:]),self.param_sym)

		# expression evaluation
		self.eval = cs.Function('eval',[syms[0],self.param_sym],[expression_]).expand()

		# collect the sx version of the symbols
		sx_syms = self.eval.sx_in()
		self.main_sym = sx_syms[0]

		# expression jacobian
		jacobian = cs.jacobian(expression_,syms[0])
		self.eval_jac = cs.Function('eval_jac',[syms[0],self.param_sym],[jacobian]).expand()

		# hessian of each of the elements of the expression
		split_eval = [cs.Function('eval',[syms[0],self.param_sym],[expression_[i]]).expand() for i in range(expression_.shape[0]) ]
		hessian = [cs.hessian(split_eval[i](*sx_syms),sx_syms[0])[0]  for i in range(expression_.shape[0])]
		self.eval_hes = [cs.Function('eval_hes'+str(i),sx_syms,[hessian[i]]).expand() for i in range(expression.shape[0])]



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

	def pack_for_julia(self,param_values,extra_info={}):
		if self.lib_path is None:
			raise NameError("CvxFuncForJulia : missing compilation step")


		param_vector = []
		for name in self.param_names:
			if type(param_values[name]) == list:
				param_vector.extend(param_values[name])
			elif type(param_values[name]) in [type(cs.DM()),type(cs.MX()),type(cs.SX())]:
				param_vector.extend(oc.cs2list(param_values[name]))
			else:
				param_vector.extend(oc.list(param_values[name]))


		# construct hessian and jacobian
		jacobian = self.eval_jac(self.main_sym,param_vector)
		jcb_nnz = jacobian.nnz()
		jcb_sparsity = jacobian.sparsity().get_triplet()

		hessian = [self.eval_hes[i](self.main_sym,param_vector) for i in range(self.sizes[1])]
		hes_nnz = [hessian[i].nnz() for i in range(self.sizes[1])]
		hes_sparsity = [hessian[i].sparsity().get_triplet() for i in range(self.sizes[1])]


		if self.sizes[1] > 1:
			pack = {'param_vector':param_vector,
					'sizes':self.sizes,
					'jcb_nnz':jcb_nnz,
					'jcb_sparsity':jcb_sparsity,
					'hes_nnz':hes_nnz,
					'hes_sparsity':hes_sparsity,
					'lib_path':self.lib_path}
		else:
			pack = {'param_vector':param_vector,
					'sizes':self.sizes,
					'grd_nnz':jcb_nnz,
					'grd_sparsity':jcb_sparsity[1],
					'hes_nnz':hes_nnz[0],
					'hes_sparsity':hes_sparsity[0],
					'lib_path':self.lib_path}

		# add possible extra information
		pack.update(extra_info)

		return pack
