from django.urls import path
from .views import PredictOverview
from .views import CorrectLabel
from .views import SummarizeNormalize

urlpatterns = [
    path("predict/", PredictOverview.as_view()),
    path("feedback/",CorrectLabel.as_view()),
    path("normal/", SummarizeNormalize.as_view()),
]
