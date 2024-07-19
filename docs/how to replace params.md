## in mnet.py

```python

        #####!!!! still not copy grad_fn
        # state_dict = self.state_dict()
        # for key, value in weights.items():
        #     state_dict[key] = value
        # self.load_state_dict(state_dict)


       with torch.no_grad():
            param_dict = resnet_model.named_parameters()
            for key, param in param_dict:
                weight = weights[key]

                param: nn.Parameter
                weight: torch.Tensor

                print(f"customed forward: key: {key} -- param: {param.shape} -- weight: {weight.shape}")

                
                #####!!!! DO capture the key in the partial function
                def _backward_hook(grad, weights=weights, key=key, state={
                    "key": key,
                    "param_original": param.data.clone().detach()
                }):
                    weight = weights[key]
                    print(f"backward_hook: {key} -- grad:{grad.shape} -- weight:{weight.shape} -- param_original:{state["param_original"].shape} -- state key: {state["key"]}")
                    weight.backward(gradient=grad, retain_graph=True)

                ####! the lambda will only capture the last value of key
                #### capture not in the define time, but in the call time
                
                # handle = param.register_hook(lambda grad: _backward_hook(grad))

                #####!!!!! must use partial instead of lambda
                handle = param.register_hook(partial(_backward_hook, weights=weights, key=key))

                # handle = param.register_hook(lambda grad: backward_hook(grad, weight, {
                #     "key": key,
                #     "param_original": param.data.clone().detach()
                # }))
                # param.register_hook(lambda grad: weight.backward(gradient=grad, retain_graph=True))

                #####! the inplace copy_ will won't copy the grad_fn!
                param.copy_(weight)

                ###!!!! can not assgin grad_fn! this is read-only on the c++ engine side
                # param.data = weight
                # param.grad_fn = weight.grad_fn
```