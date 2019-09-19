# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name,arguments-differ,no-else-return,unused-argument,missing-docstring
from __future__ import absolute_import
from ..expr_functor import ExprMutator
from ..expr import Function, Call, Let, Var, GlobalVar, Constant, Tuple, TupleGetItem
from .. import transform as _transform
from .. import module as _module
from .. import cast
"""
Relay Downcast from Full-precision to Half-precision floating-point Pass
"""

def downcast_fp16(func):
    # get_valid_counts and non_max_suppression does not support fp16 so we create a filter list for them
    filter_list = ['vision.get_valid_counts', 'vision.non_max_suppression']
    """
    Parameters
    ---------
    graph: Function
        The original graph.

    Retruns
    -------
    The graph after dowmcasting to half-precision floating-point.
    """
    class DowncastMutator(ExprMutator):
        def visit_call(self, call):
            dtype = 'float16'
            if call.op.name in filter_list:
                dtype = 'float32'
            new_fn = self.visit(call.op)
            # Collec the original dtypes
            type_list = []
            if call.op.name in filter_list:
                # For nms
                for arg in call.args:
                    if isinstance(arg, TupleGetItem):
                        arg_tuple = arg.tuple_value
                        if isinstance(arg_tuple, Call):
                            tuple_types = arg_tuple.checked_type.fields
                            type_list.append(tuple_types[arg.index].dtype)
                if call.op.name == 'vision.get_valid_counts':
                    tuple_types = call.checked_type.fields
                    for cur_type in tuple_types:
                        type_list.append(cur_type.dtype)

            args = [self.visit(arg) for arg in call.args]
            new_args = list()
            arg_idx = 0
            for arg in args:
                if isinstance(arg, Var) or isinstance(arg, Constant):
                    new_args.append(cast(arg, dtype=dtype))
                else:
                    if call.op.name in filter_list:
                        if isinstance(arg, TupleGetItem):
                            if type_list[arg_idx] == 'int32':
                                new_args.append(arg)
                            else:
                                new_args.append(cast(arg, dtype=dtype))
                        else:
                            new_args.append(cast(arg, dtype=dtype))
                    else:
                        new_args.append(arg)
                arg_idx += 1
            if call.op.name in filter_list:
                if call.op.name == 'vision.get_valid_counts':
                    return Call(new_fn, new_args, call.attrs)
                else:
                    return cast(Call(new_fn, new_args, call.attrs), dtype='float16')
            else:
                return Call(new_fn, new_args, call.attrs)

    class UpcastMutator(ExprMutator):
        def visit_call(self, call):
            return cast(call, dtype='float32')

    def infer_type(expr):
        mod = _module.Module.from_expr(expr)
        mod = _transform.InferType()(mod)
        entry = mod["main"]
        return entry if isinstance(expr, Function) else entry.body

    func = infer_type(func)
    downcast_pass = DowncastMutator()
    func = downcast_pass.visit(func)
    upcast_pass = UpcastMutator()
    func = upcast_pass.visit(func)
    func = infer_type(func)
    return func

