import pytest

import numpy as np

from cascade_at.core.form.fields import (
    Form,
    FormList,
    IntField,
    OptionField,
    StringListField,
    ListField,
    Dummy,
)


@pytest.fixture
def nested_forms():
    class EvenMoreInner(Form):
        int_list_field = StringListField(separator=" ", constructor=float)

    class MyInnerSanctum(Form):
        int_field = IntField()
        float_list_field = StringListField(separator=" ", constructor=float)
        further_in = FormList(EvenMoreInner)

    class MyInnerForm(Form):
        inner_sanctum = MyInnerSanctum()
        nothing_here_either = Dummy()

    class MyOuterForm(Form):
        inner_form = MyInnerForm()
        nothing_here = Dummy()

    return MyOuterForm


@pytest.fixture
def form_with_form_list():
    class MyInnerForm(Form):
        int_field = IntField()

    class MyOuterForm(Form):
        inner_forms = FormList(MyInnerForm)

    return MyOuterForm


@pytest.fixture
def form_with_options():
    class MyForm(Form):
        str_option_field = OptionField(["a", "b", "see"])
        int_option_field = OptionField([1, 2, 3], constructor=int)

    return MyForm


@pytest.fixture
def form_with_string_list():
    class MyForm(Form):
        ints_field = StringListField(separator=" ", constructor=int)

    return MyForm


@pytest.fixture
def form_with_string_list_commas():
    class MyForm(Form):
        ints_field = StringListField(separator=",", constructor=int)

    return MyForm


@pytest.fixture
def form_with_list():
    class MyForm(Form):
        floats_field = ListField(separator=" ", constructor=float)

    return MyForm


def test_nested_forms__succesful_validation(nested_forms):
    f = nested_forms(
        {
            "inner_form": {
                "inner_sanctum": {
                    "int_field": "10",
                    "float_list_field": "10 5.5 nan",
                    "further_in": [{"int_list_field": "1 2 3"}],
                }
            }
        }
    )
    assert not f.validate_and_normalize()


def test_nested_forms__failed_validation(nested_forms):
    f = nested_forms(
        {
            "inner_form": {
                "inner_sanctum": {
                    "int_field": "oeueou",
                    "float_list_field": "10 5.5 non",
                    "further_in": [{"int_list_field": "1 2 3"}],
                }
            }
        }
    )
    assert set(f.validate_and_normalize()) == {
        ("inner_form.inner_sanctum.int_field", "inner_form.inner_sanctum.int_field",
         "Invalid int value 'oeueou'"),
        ("inner_form.inner_sanctum.float_list_field", "inner_form.inner_sanctum.float_list_field",
         "Errors in items: [Invalid float value 'non']"),
    }


def test_nested_forms__normalization(nested_forms):
    f = nested_forms(
        {
            "inner_form": {
                "inner_sanctum": {
                    "int_field": "10",
                    "float_list_field": "10 5.5 nan",
                    "further_in": [{"int_list_field": "1 2 3"}],
                }
            }
        }
    )
    f.validate_and_normalize()
    assert f.inner_form.inner_sanctum.int_field == 10
    assert f.inner_form.inner_sanctum.float_list_field[:2] == [10.0, 5.5]
    assert np.isnan(f.inner_form.inner_sanctum.float_list_field[-1])
    assert len(f.inner_form.inner_sanctum.float_list_field) == 3
    assert f.inner_form.inner_sanctum.further_in[0].int_list_field == [1, 2, 3]


def test_form_with_FormList__empty(form_with_form_list):
    f = form_with_form_list({"inner_forms": []})
    assert not f.validate_and_normalize()
    assert len(f.inner_forms) == 0


def test_form_with_FormList__non_empty(form_with_form_list):
    f = form_with_form_list({"inner_forms": [{"int_field": "10"}, {"int_field": "20"}]})
    assert not f.validate_and_normalize()
    assert f.inner_forms[0].int_field == 10
    assert f.inner_forms[1].int_field == 20


def test_form_with_FormList__to_dict(form_with_form_list):
    f = form_with_form_list({"inner_forms": [{"int_field": "10"}, {"int_field": "20"}]})
    f.validate_and_normalize()
    assert f.to_dict() == {"inner_forms": [{"int_field": 10}, {"int_field": 20}]}


def test_form_with_FormList__validation(form_with_form_list):
    f = form_with_form_list({"inner_forms": [{"int_field": "oeueoueo"}, {"int_field": "20"}]})
    assert f.validate_and_normalize() == [
        ("inner_forms[0].int_field", "inner_forms[0].int_field", "Invalid int value 'oeueoueo'")
    ]


def test_OptionField__success(form_with_options):
    f = form_with_options({"str_option_field": "a", "int_option_field": "1"})
    assert not f.validate_and_normalize()


def test_OptionField__validation(form_with_options):
    f = form_with_options({"str_option_field": "c", "int_option_field": "oueo"})
    assert set(f.validate_and_normalize()) == {
        ("str_option_field", "str_option_field", "Invalid option 'c'"),
        ("int_option_field", "int_option_field", "Invalid int value 'oueo'"),
    }


def test_OptionField__normalization(form_with_options):
    f = form_with_options({"str_option_field": "a", "int_option_field": "3"})
    f.validate_and_normalize()
    assert f.str_option_field == "a"
    assert f.int_option_field == 3


def test_ListField__successful_validation(form_with_list):
    f = form_with_list({"floats_field": ["1", "2", "3", "4", "5"]})
    assert not f.validate_and_normalize()


def test_ListField__successful_normalization(form_with_list):
    f = form_with_list({"floats_field": ["1", "2", "3", "4", "5"]})
    f.validate_and_normalize()
    assert f.floats_field == [1, 2, 3, 4, 5]


def test_ListField__failed_validation(form_with_list):
    f = form_with_list({"floats_field": ["1", "2", "three", "4", "five"]})
    assert f.validate_and_normalize() == [
        ("floats_field", "floats_field", "Errors in items: [Invalid float value 'three', Invalid float value 'five']")
    ]


def test_StringListField__successful_validation(form_with_string_list):
    f = form_with_string_list({"ints_field": "1 2 3 4 5"})
    assert not f.validate_and_normalize()


def test_StringListField__failed_validation(form_with_string_list):
    f = form_with_string_list({"ints_field": ["1 2 3 4 5"]})
    assert f.validate_and_normalize() == [
        ('ints_field', 'ints_field', "Errors in items: [Invalid int value '['1 2 3 4 5']']"),
    ]


def test_StringListField__to_dict(form_with_string_list):
    f = form_with_string_list({"ints_field": "1 2 3 4 5"})
    f.validate_and_normalize()
    assert f.to_dict() == {"ints_field": "1 2 3 4 5"}


@pytest.mark.parametrize('string_input', ["1 2 3 4 5", " 1   2 3 4  5",
                                          "1  2   3   4 5    "])
def test_StringListField__success_bad_input(form_with_string_list,
                                            string_input):
    f = form_with_string_list({"ints_field": string_input})
    f.validate_and_normalize()
    assert f.ints_field == [1, 2, 3, 4, 5]


@pytest.mark.parametrize('string_input', ["1,2,3,4,5", "1, 2, 3, 4, 5",
                                          " 1,2, 3,  4,  5 ",
                                          "  1,,,2,, 3  ,,,4,5"])
def test_StringListField__success_commas(form_with_string_list_commas,
                                         string_input):
    f = form_with_string_list_commas({"ints_field": string_input})
    f.validate_and_normalize()
    assert f.ints_field == [1, 2, 3, 4, 5]
