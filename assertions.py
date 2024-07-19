from typing_extensions import Callable

def default_error_template_fun(values: list[tuple[str,int]]) -> str:
    values_str = '\n'.join(map(
        lambda expr_value: f"{expr_value[0]} = {expr_value[1]}",
        values
    ))
    return f"Inconsistent values found:\n{values_str}"

def assert_equal(
    exprs: list[str],
    locals: dict,
    error_template_fun: Callable[[list[tuple[str,int]]], str] = default_error_template_fun
):
    values_to_check: list[tuple[str,int]] = []
    for expr in exprs:
        try:
            values_to_check.append((expr, eval(expr, None, locals)))
        except:
            pass
        
    common_value = None
    for (expr, value) in values_to_check:
        if common_value is None:
            common_value = value
        else:
            if value != common_value:
                raise Exception(error_template_fun(values_to_check))
