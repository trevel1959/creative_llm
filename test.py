from string import Template

template = Template("Hello, $name! Today is $day.")
formatted_string = template.safe_substitute(name="Alice")
print(formatted_string)
formatted_string = template.safe_substitute(day="monday")
print(formatted_string)