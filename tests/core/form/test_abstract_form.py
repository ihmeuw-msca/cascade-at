import pytest

from cascade_at.core.form.abstract_form import Form, Field, SimpleTypeField


def test_development_target():
    class IntField(Field):
        def _validate_and_normalize(self, instance, value):
            try:
                new_value = int(value)
            except (ValueError, TypeError):
                return None, f"Invalid integer value '{value}'"
            return new_value, None

    class MyInnerForm(Form):
        my_inner_field = IntField(display="My Inner Field!!")

    class MyForm(Form):
        my_field = IntField()
        my_inner_form = MyInnerForm()

    f = MyForm({"my_field": "10", "my_inner_form": {"my_inner_field": 100}})
    errors = f.validate_and_normalize()
    assert not errors

    assert f.my_field == 10
    assert f.my_inner_form.my_inner_field == 100

    f = MyForm({"my_field": "10", "my_inner_form": {"my_inner_field": "eaoeao"}})
    errors = f.validate_and_normalize()
    assert set(errors) == {
        ("my_inner_form.my_inner_field", "my_inner_form.My Inner Field!!", "Invalid integer value 'eaoeao'")
    }

    f = MyForm({"my_field": None, "my_inner_form": {}})
    errors = f.validate_and_normalize()
    assert set(errors) == {
        ("my_field", "my_field", "Invalid integer value 'None'"),
        ("my_inner_form.my_inner_field", "my_inner_form.My Inner Field!!", "Missing data"),
    }


def test_Field__is_unset():
    class MyForm(Form):
        my_field = SimpleTypeField(int)

    f = MyForm({"my_field": 10})
    assert not type(f).__dict__["my_field"].is_unset(f)

    f = MyForm({"my_field": 0})
    assert not type(f).__dict__["my_field"].is_unset(f)

    f = MyForm({"my_field": None})
    assert not type(f).__dict__["my_field"].is_unset(f)

    f = MyForm({})
    assert type(f).__dict__["my_field"].is_unset(f)


def test_Form__is_unset():
    class MyForm(Form):
        my_field = SimpleTypeField(int)
        my_other_field = SimpleTypeField(int)

    f = MyForm({"my_field": 10, "my_other_field": 5})
    assert not f.is_unset()

    f = MyForm({"my_field": 10})
    assert not f.is_unset()

    f = MyForm({})
    assert f.is_unset()


def test_Field__nullable():
    class MyForm(Form):
        my_field = SimpleTypeField(int)
        my_other_field = SimpleTypeField(int, nullable=True)

    f = MyForm({"my_field": 10, "my_other_field": 5})
    assert not f.validate_and_normalize()

    f = MyForm({"my_field": 10})
    assert not f.validate_and_normalize()


def test_Form__nullable():
    class MyInnerForm(Form):
        my_field = SimpleTypeField(int)
        my_other_field = SimpleTypeField(int)

    class MyForm(Form):
        inner = MyInnerForm(nullable=True)

    f = MyForm({"inner": {"my_field": 10, "my_other_field": 0}})
    assert not f.validate_and_normalize()

    f = MyForm({"inner": {}})
    assert not f.validate_and_normalize()


def test_Field__default_value():
    class MyForm(Form):
        my_field = SimpleTypeField(int, nullable=True)
        my_other_field = SimpleTypeField(int, nullable=True, default=float("inf"))

    f = MyForm({})
    f.validate_and_normalize()
    assert f.my_field is None
    assert f.my_other_field == float("inf")


def test_Form__name_field():
    class MyInnerForm(Form):
        my_field = SimpleTypeField(int)
        foo = SimpleTypeField(str)

    class MyOuter(Form):
        inner_form = MyInnerForm(name_field="foo")

    f = MyOuter({"inner_form": {"my_field": "10", "foo": "not_inner_form"}})
    assert not f.validate_and_normalize()
    assert f.inner_form.foo == "inner_form"


def test_Form__display_name():
    class MyInnerForm(Form):
        my_field = SimpleTypeField(int, display="Cheesy Interior")

    class MyOuter(Form):
        inner_form = MyInnerForm(display="Flaky Crust")

    f = MyOuter({"inner_form": {"my_field": "abc"}})
    assert set(f.validate_and_normalize()) == {
        ("inner_form.my_field", "Flaky Crust.Cheesy Interior", "Invalid int value 'abc'")
    }


def test_SimpleTypeField__validation():
    class MyForm(Form):
        my_int_field = SimpleTypeField(int)
        my_float_field = SimpleTypeField(float)

    f = MyForm({"my_int_field": 10, "my_float_field": 1.5})
    errors = f.validate_and_normalize()
    assert not errors

    f = MyForm({"my_int_field": "oueou", "my_float_field": "blaa"})
    errors = f.validate_and_normalize()
    assert set(errors) == {
        ("my_int_field", "my_int_field", "Invalid int value 'oueou'"),
        ("my_float_field", "my_float_field", "Invalid float value 'blaa'"),
    }


def test_SimpleTypeField__normalization():
    class MyForm(Form):
        my_int_field = SimpleTypeField(int)
        my_float_field = SimpleTypeField(float)

    f = MyForm({"my_int_field": 10, "my_float_field": 1.5})
    f.validate_and_normalize()
    assert f.my_int_field == 10
    assert f.my_float_field == 1.5


@pytest.fixture
def form_with_validation():
    class MyForm(Form):
        a = SimpleTypeField(int)
        b = SimpleTypeField(int)
        c = SimpleTypeField(int)

        def _full_form_validation(self, root):
            if self.a < self.b:
                return ["a must be >= b"]
            return []

    return MyForm


def test_full_form_validation__successful(form_with_validation):
    f = form_with_validation({"a": "10", "b": 8, "c": 1})
    assert not f.validate_and_normalize()


def test_full_form_validation__failure_due_to_form_validation(form_with_validation):
    f = form_with_validation({"a": "1", "b": 8, "c": 1})
    assert f.validate_and_normalize() == [("", "", "a must be >= b")]


def test_full_form_validation__failure_due_to_inconsistent_state(form_with_validation):
    f = form_with_validation({"a": "1", "b": 8, "c": "ooeueou"})
    errors = f.validate_and_normalize()
    assert errors
    assert ("", "", "a must be >= b") not in errors


def test_subform_with_defaults():
    class MyInnerForm(Form):
        a = SimpleTypeField(int, default=42, nullable=True)

    class MyForm(Form):
        inner = MyInnerForm()

    f = MyForm({})
    errors = f.validate_and_normalize()
    assert not errors
    assert f.inner.a == 42


def test_validation_priority():
    class InnerOne(Form):
        a = SimpleTypeField(int)

        def _full_form_validation(self, root):
            assert isinstance(root.two.b, int)
            return []

    class InnerTwo(Form):
        b = SimpleTypeField(int, validation_priority=5)

        def _full_form_validation(self, root):
            assert isinstance(root.one.a, str)
            return []

    class MyForm(Form):
        one = InnerOne()
        two = InnerTwo()

    f = MyForm({"one": {"a": "10"}, "two": {"b": "15"}})
    f.validate_and_normalize()


def test_to_dict():
    class MyInnerForm(Form):
        b = SimpleTypeField(str)
        c = SimpleTypeField(int, default=42, nullable=True)

    class MyForm(Form):
        a = SimpleTypeField(float)
        inner = MyInnerForm()

    f = MyForm({"a": 1.5, "inner": {"b": "test"}})
    f.validate_and_normalize()

    result = f.to_dict()

    assert result == {"a": 1.5, "inner": {"b": "test"}}
