{{ fullname | escape | underline}}

{# :no-members: overrides autodoc_default_options["members"] for the module page only. Without it
   the module page documents every function and class inline AND the per-object stub pages below
   document them again, which Sphinx reports as "duplicate object description" (~1000 of them).
   The rubrics below still summarize and link to the stub pages, so nothing becomes unreachable. #}
.. automodule:: {{ fullname }}
    :no-members:

    {% block attributes %}
    {% if attributes %}
    .. rubric:: Module Attributes

    .. autosummary::
        :toctree:
    {% for item in attributes %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block functions %}
    {% if functions %}
    .. rubric:: {{ _('Functions') }}

    .. autosummary::
        :toctree:
    {% for item in functions %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block classes %}
    {% if classes %}
    .. rubric:: {{ _('Classes') }}

    .. autosummary::
        :template: autosummary_class.rst
        :toctree:
    {% for item in classes %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block exceptions %}
    {% if exceptions %}
    .. rubric:: {{ _('Exceptions') }}

    .. autosummary::
        :toctree:
    {% for item in exceptions %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
    :toctree:
    :template: autosummary.rst
    :recursive:
{% for item in modules %}
    {% if item not in excluded_modules %}
        {{ item }}
    {%- endif %}
{%- endfor %}
{% endif %}
{% endblock %}