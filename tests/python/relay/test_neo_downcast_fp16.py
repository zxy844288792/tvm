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

import numpy as np
import tvm
from tvm import relay
from tvm.relay.testing import resnet
import time

def test_downcast_fp16_resnet():
    image_shape = (3,224,224)
    net, params = resnet.get_workload(image_shape=image_shape, dtype='float32')
    # float32 
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(net, params=params, target='llvm')
    
    # float16
    func = relay.neo.downcast_fp16(net['main'])
    with relay.build_config(opt_level=3):
        graph_fp16, lib_fp16, params_fp16 = relay.build(net, params=params, target='llvm')
    
    rt = tvm.contrib.graph_runtime.create(graph, lib, tvm.cpu())
    rt.set_input(**params)
    rt_fp16 = tvm.contrib.graph_runtime.create(graph_fp16, lib_fp16, tvm.cpu())
    rt_fp16.set_input(**params_fp16)
    for i in range(100):
        X = tvm.nd.array(np.random.random_sample(image_shape).astype('float32'))
        rt.set_input('data', X)
        rt_fp16.set_input('data', X)
        
        rt.run()
        rt_fp16.run()
        out = rt.get_output(0).asnumpy()
        out_fp16 = rt.get_output(0).asnumpy()
        np.testing.assert_equal(np.argmax(out, axis=1), np.argmax(out_fp16, axis=1))

if __name__ == "__main__":
    test_downcast_fp16_resnet()
