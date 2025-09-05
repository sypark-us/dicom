"""
Defines the functions implementing the AI-rank pipeline.
"""
import gc
import threading
from   threading import Lock, Event
from   datetime  import date, datetime, time, timedelta, timezone
from   typing    import Any, List, Optional, Tuple

import aiw.logger as log

from   aiw.pipeline.rank.types import UserIdMapping
from   aiw.pipeline.rank.types import SpecialtyMapping
from   aiw.pipeline.rank.types import RankModelResources
from   aiw.pipeline.rank.types import RankPipelineInputs
from   aiw.pipeline.rank.types import RankPipelineOutputs
from   aiw.pipeline.rank.types import RankPipelineConfiguration
from   aiw.pipeline.rank.types import RankPipelineUpdateDispositionInputs

from   aiw.pipeline.models import AlgorithmNames
from   aiw.pipeline.models import ModelNames
from   aiw.pipeline.models import ModelState
from   aiw.pipeline.models import ModelDescriptor
from   aiw.pipeline.models import ModelRepository
from   aiw.pipeline.models import model_state_to_str
from   aiw.pipeline.models import is_newer_model_version

from   aiw.pipeline.rank.compare_models import ModelComparison

from   aiw.pipeline.rank.algorithms.rank_study import RankStudyModel


_MIN_TIME_BETWEEN_UPDATES_SECONDS  : int = 10


class AIWorklistPipelineConfiguration:
    """
    Information used to configure an `AIWorklistPipeline` instance.

    Fields
    ------
        model_repository    : The `ModelRepository` managing the model(s) that comprise the algorithm pipeline.
        train_models        : Specify `True` if the algorithm pipeline should perform online model training (if any models support online training).
        enable_randomization: Specify `True` to enable randomization within the algorithm pipeline (if any models perform randomization), or `False` to disable randomization to force deterministic behavior.
        update_interval     : The minimum time that must elapse between model resource loads, in seconds. Specify 0 to not throttle loading, or a negative value to use the default time.
    """
    def __init__(self, repo: ModelRepository, train_models: bool, enable_randomization: bool=True, update_interval: int=_MIN_TIME_BETWEEN_UPDATES_SECONDS) -> None:
        if repo is None or not repo.repository_root or not repo.distribution_root:
            raise ValueError('A valid, initialized ModelRepository must be specified')
        if update_interval < 0 or (update_interval != 0 and update_interval < _MIN_TIME_BETWEEN_UPDATES_SECONDS):
            update_interval = _MIN_TIME_BETWEEN_UPDATES_SECONDS
        
        self.model_repository    : ModelRepository = repo
        self.train_models        : bool            = train_models
        self.enable_randomization: bool            = enable_randomization
        self.update_interval     : int             = update_interval



class AIWorklistPipelinePerformanceData:
    """
    Data used to analyze the current performance of the pipeline.
    """
    def __init__(self) -> None:
        self.total_reward : float = 0.0
        self.num_rewards  : int   = 0
        self.correct_count: int   = 0
        self.penalty_count: int   = 0
        self.total_loss   : float = 0.0
        self.num_loss     : float = 0.0




class AIWorklistPipeline:
    """
    Implements the main interface to the AI worklist algorithm pipeline.

    Fields
    ------
        model_repository        : The `ModelRepository` used to manage models associated with the pipeline.
        train_models            : Indicates whether the algorithm pipeline is configured to perform online model training (if any models support online training).
        enable_randomization    : Indicates whether randomization is enabled within the algorithm pipeline (if any models perform randomization).
        resource_update_interval: The minimum time that must elapse between pipeline model resource loads, in seconds. This value will be 0 if resource loading is not throttled.
    """

    def __init__(self, pipeline_config: AIWorklistPipelineConfiguration) -> None:
        if pipeline_config is None:
            raise ValueError('A valid AIWorklistPipelineConfiguration must be specified')

        self.model_repository         : ModelRepository    = pipeline_config.model_repository
        self.train_models             : bool               = pipeline_config.train_models
        self.enable_randomization     : bool               = pipeline_config.enable_randomization
        self.resource_update_interval : int                = pipeline_config.update_interval
        self._rank_model              : RankStudyModel     = None
        self._rank_model_resources    : RankModelResources = None
        self._pending_model_updates   : bool               = False
        self._resource_loader_lock    : Lock               = Lock()
        self._performance_eval_lock   : Lock               = Lock()
        self._last_load_time          : datetime           = None
        self._last_eval_time          : datetime           = None
        self._allow_perf_eval         : bool               = True
        self.config_perf_eval         : bool               = False


    def is_executable(self) -> bool:
        """
        Check whether operations other than update checking and resource loading can be issued against the pipeline.

        Return
        ------
            `True` if the pipeline instance is available for use.
        """
        ok1: bool = False
        ok2: bool = False

        with self._resource_loader_lock as _:
            ok1 = False if not self._rank_model_resources else self._rank_model_resources.resident_in_memory()
            ok2 = False if not self._rank_model else True
            # ...Other pipeline resource conditions to check...
        
        return all((ok1, ok2))


    def load_resources(self, force_load: bool=False) -> None:
        """
        Attempt to load model resources, configure the pipeline, and prepare the pipeline for use.

        Parameters
        ----------
            force_load: Specify `True` to force the loading of resources even though the minimum update interval has not elapsed.
        """
        repo    : ModelRepository = self.model_repository
        utc_now : datetime        = datetime.now(timezone.utc)
        did_load: bool            = False
        elapsed : int             = 0

        with self._resource_loader_lock as _:
            # If resources have never been loaded, force a load.
            if self._last_load_time is None:
                self._last_load_time = utc_now
                force_load = True

            # Calculate the number of seconds elapsed since the prior load.
            elapsed = int((utc_now - self._last_load_time).total_seconds())
            if not force_load and elapsed < self.resource_update_interval:
                log.debug(f'Skipping load of {AlgorithmNames.AI_WORKLIST} pipeline resources; {elapsed} seconds have elapsed since previous load and configured update interval set to {self.resource_update_interval} seconds.')
                return

            try: # Pipeline resources should be (re)loaded.
                log.info(f'Retrieving latest model information for {AlgorithmNames.AI_WORKLIST} pipeline models.')
                model_state    : ModelState                = ModelState.TRAINING if self.train_models else ModelState.FROZEN
                rank_model_desc: ModelDescriptor           = None
                rank_resources : RankModelResources        = None
                rank_config    : RankPipelineConfiguration = None
                rank_model     : RankStudyModel            = None
                algo_model_list: List[ModelDescriptor]     = repo.get_latest_models_for_algorithm(AlgorithmNames.AI_WORKLIST, model_state)
                for model_info in algo_model_list:
                    if model_info.model_name == ModelNames.AI_WORKLIST_RANK:
                        rank_model_desc = model_info
                    # ...others, eventually.
                if rank_model_desc is None:
                    log.error(f'No model {ModelNames.AI_WORKLIST_RANK} for algorithm pipeline {AlgorithmNames.AI_WORKLIST} in state {model_state_to_str(model_state)} could be found.')
                    return

                log.info(f'The {AlgorithmNames.AI_WORKLIST} pipeline will attempt to use model {rank_model_desc.get_short_model_name()}:')
                log.info(f'- Architecture revision: {rank_model_desc.architecture_version}')
                log.info(f'- Model revision       : {rank_model_desc.model_revision}')
                log.info(f'- Model state          : {rank_model_desc.model_state}')
                log.info(f'- Component count      : {len(rank_model_desc)}')
                for model_component in rank_model_desc:
                    log.info(f'--- {model_component.data_type}      : {model_component.get_full_model_name()} @ {repo.repository_root}')

                log.info(f'Attempting to load {AlgorithmNames.AI_WORKLIST} pipeline model resources.')
                rank_resources = RankModelResources(repo, rank_model_desc)
                if not rank_resources.try_load_resources():
                    log.error(f'One or more errors occurred while trying to load resources for the {AlgorithmNames.AI_WORKLIST} model {rank_model_desc.get_short_model_name()}.')
                    return

                log.info(f'Initializing {AlgorithmNames.AI_WORKLIST} pipeline model {rank_model_desc.get_short_model_name()}.')
                rank_config= RankPipelineConfiguration(rank_resources, train_model=self.train_models, randomize_rows=self.enable_randomization)
                rank_model = RankStudyModel(rank_config)
                log.info(f'- Configuration: Feature Length={rank_config.feature_length}')
                log.info(f'- Configuration: Maximum Candidates per-Request={rank_config.max_candidates}')
                log.info(f'- Configuration: Maximum Specialty Identifiers={rank_config.max_specialties}')
                log.info(f'- Configuration: Balance Vector Length={rank_config.balance_feature_length}')
                log.info(f'- Configuration: Recent Exam Count={rank_config.recent_exam_count}')
                log.info(f'- Configuration: Active User Timeout (sec)={rank_config.active_user_timeout}')
                log.info(f'- Configuration: Performance Data Count={rank_config.performance_data_count}')
                log.info(f'- Configuration: Randomize Rows={rank_config.randomize_rows}')
                log.info(f'- Configuration: Train Model={rank_config.train_model}')
                log.info(f'- Completed initialization for {AlgorithmNames.AI_WORKLIST} pipeline.')

                # Future operations will use the updated model(s).
                # Note: Don't want to unload the current resources, if any.
                # - There could be active operations still referencing them.
                self._rank_model            = rank_model
                self._rank_model_resources  = rank_resources
                self._pending_model_updates = False
                self._last_load_time        = datetime.now(timezone.utc)
                self.config_perf_eval       = rank_config.performance_eval


                # Indicate that the load completed successfully.
                did_load = True

            except Exception as _: # pylint: disable=broad-except
                log.exception(f'Something went wrong during resource loading for the {AlgorithmNames.AI_WORKLIST} pipeline.')
        
        # If a load just completed successfully, the model is 'fresh' and attempts to execute
        # a performance evaluation should be ignored.
        with self._performance_eval_lock as _:
            if did_load:
                self._allow_perf_eval = False
    

    def check_for_updates(self, force_refresh: bool=False) -> bool:
        """
        Check for a newer version of any model associated with the AI-worklist pipeline.

        Parameters
        ----------
            force_refresh: Specify `True` to force a refresh even if the update interval has not elapsed.

        Return
        ------
            `True` if the `ModelRepository` reports that an updated version of one or more models is available for the AI-worklist pipeline.
        """
        repo   : ModelRepository = self.model_repository
        updates_available : bool = False

        repo.refresh_repository(force_refresh, report_new_states=False, update_interval_sec=self.resource_update_interval)
        # See if updates are available for any models used by the AI-worklist pipeline.
        model_state    : ModelState            = ModelState.TRAINING if self.train_models else ModelState.FROZEN
        rank_model_new : ModelDescriptor       = None
        rank_model_curr: ModelDescriptor       = None
        algo_model_list: List[ModelDescriptor] = repo.get_latest_models_for_algorithm(AlgorithmNames.AI_WORKLIST, model_state)
        for model_info in algo_model_list:
            if model_info.model_name == ModelNames.AI_WORKLIST_RANK:
                rank_model_new = model_info
            # ...Others, eventually.

        with self._resource_loader_lock as _:
            # Note: It's possible that the pipeline's load_resources function hasn't returned yet, or has failed.
            rank_model_curr    = None if self._rank_model_resources is None else self._rank_model_resources.model_descriptor
            rank_model_updated = is_newer_model_version(rank_model_new, rank_model_curr)
            # ...Others, eventually...
            if any((rank_model_updated,)): # Any model(s) updated during this check?
                self._pending_model_updates = True
                updates_available = True
            if not updates_available:      # Any model(s) updated during a prior check, that haven't been loaded yet?
                updates_available = self._pending_model_updates

        return updates_available
    

    def rank_candidates(self, inputs: RankPipelineInputs) -> Optional[RankPipelineOutputs]:
        """
        Rank candidates in decreasing order of preference for having a specific study assigned for reading.

        Parameters
        ----------
            inputs: Information about the study and candidates to be ranked for assignment.
        
        Return
        ------
            A `RankPipelineOutputs` specifying the ranked candidate list. Note that the set of output candidates may be empty.
        """
        if inputs is None:
            log.error(f'No RankPipelineInputs specified in call to {AlgorithmNames.AI_WORKLIST} model {ModelNames.AI_WORKLIST_RANK} rank_candidates.')
            return RankPipelineOutputs()
        if inputs.work_item is None:
            log.error(f'No input Exam specified in inputs to {AlgorithmNames.AI_WORKLIST} model {ModelNames.AI_WORKLIST_RANK} rank_candidates.')
            return RankPipelineOutputs()
        if not inputs.candidates:
            log.error(f'No candidate Radiologists specified in inputs to {AlgorithmNames.AI_WORKLIST} model {ModelNames.AI_WORKLIST_RANK} rank_candidates.')
            return RankPipelineOutputs()

        # Retrieve the rank model in use at the moment of the call.
        rank_model: RankStudyModel = None
        with self._resource_loader_lock as _:
            rank_model = self._rank_model

        if rank_model is None:
            log.error(f'No RankStudyModel is available in {AlgorithmNames.AI_WORKLIST} pipeline; did resource loading fail [rank_candidates]?')
            return RankPipelineOutputs()

        outputs = RankPipelineOutputs()
        results : List[Tuple[int, float, float]] = rank_model.get_ranking(inputs.work_item, inputs.candidates)
        if results: # NOTE: Length of results may be less than length of candidates.
            outputs.ranked_candidate_list = results
        
        # If this pipeline is set to train, then a model with improved performance is potentially available.
        with self._performance_eval_lock as _:
            if self.train_models:
                self._allow_perf_eval = True

        return outputs
    

    def update_work_item_state(self, update: RankPipelineUpdateDispositionInputs) -> None:
        """
        Update the candidate ranking model based on the actual outcome for a specific study.

        Parameters
        ----------
            update: Data regarding the updated state of the work item and a candidate radiologist.
        """
        if update is None:
            log.error(f'No RankPipelineUpdateDispositionInputs specified in call to {AlgorithmNames.AI_WORKLIST} model {ModelNames.AI_WORKLIST_RANK} update_work_item_state.')
            return
        if update.work_item is None:
            log.error(f'No Exam specified in inputs to {AlgorithmNames.AI_WORKLIST} model {ModelNames.AI_WORKLIST_RANK} update_work_item_state.')
            return
        if update.candidate is None:
            log.error(f'No candidate Radiologist specified in inputs to {AlgorithmNames.AI_WORKLIST} model {ModelNames.AI_WORKLIST_RANK} update_work_item_state.')
            return
        if not update.disposition:
            log.error(f'No work item disposition specified in inputs to {AlgorithmNames.AI_WORKLIST} model {ModelNames.AI_WORKLIST_RANK} update_work_item_state.')
            return
            
        # Retrieve the rank model in use at the moment of the call.
        rank_model: RankStudyModel = None
        with self._resource_loader_lock as _:
            rank_model = self._rank_model

        if rank_model is None:
            log.error(f'No RankStudyModel is available in {AlgorithmNames.AI_WORKLIST} pipeline; did resource loading fail [update_work_item_state]?')
            return

        disposition: str  = update.disposition.lower()
        rejection  : bool = True if disposition == 'rejected' else False
        completed  : bool = True if disposition == 'completed' else False
        assigned   : bool = True if disposition == 'assigned' else False
        reassigned : bool = True if disposition == 'reassigned' else False

        if completed and not update.work_time:
            log.warning(f'Expected non-zero work time for completed work item {update.work_item.eid}.')

        rank_model.update_disposition(update.work_item, update.candidate, rejection, completed, assigned, reassigned, update.work_time)
        
        # If this pipeline is set to train, then a model with improved performance is potentially available.
        with self._performance_eval_lock as _:
            if self.train_models:
                self._allow_perf_eval = True


    def evaluate_training_performance(self) -> None:
        """
        Compare the performance of the pipeline model(s) under training with the performance of the prior version of the same model(s).
        If the performance of the newer model is better, create a new model version based on the newer model.
        This function may be called from a background thread since it may take several seconds to complete.
        """
        if not self.train_models:
            log.debug(f'Algorithm pipeline {AlgorithmNames.AI_WORKLIST} is not configured for model training and cannot evaluate performance.')
            return
        
        # Update the UTC timestamp of the most recent performance evaluation.
        run_evaluation : bool = self.config_perf_eval
        with self._performance_eval_lock as _:
            if not self._allow_perf_eval:
                run_evaluation = False
            if run_evaluation:
                self._last_eval_time = datetime.now(timezone.utc)

        if not run_evaluation:
            log.debug(f'Algorithm pipeline {AlgorithmNames.AI_WORKLIST} will skip performance evaluation since no model training has been performed.')
            return

        baseline_data  : Any                   = None
        repo           : ModelRepository       = self.model_repository
        frozen_models  : List[ModelDescriptor] = repo.get_latest_models_for_algorithm(AlgorithmNames.AI_WORKLIST, ModelState.FROZEN)
        training_models: List[ModelDescriptor] = repo.get_latest_models_for_algorithm(AlgorithmNames.AI_WORKLIST, ModelState.TRAINING)
        if not frozen_models:
            log.error(f'No model(s) found for {AlgorithmNames.AI_WORKLIST} pipeline in state FROZEN. Performance evaluation cannot proceed.')
            return
        if not training_models:
            log.error(f'No model(s) found for {AlgorithmNames.AI_WORKLIST} pipeline in state TRAINING. Performance evaluation cannot proceed.')
            return

        # Retrieve the current set of baseline performance data for the study ranking model.
        with self._resource_loader_lock as _:
            baseline_data = self._rank_model_resources.performance_data
        if baseline_data is None:
            log.error(f'No performance comparison baseline data for {AlgorithmNames.AI_WORKLIST} model {ModelNames.AI_WORKLIST_RANK}. Performance evaluation cannot proceed.')
            return

        # Find the newest study ranking model descriptor for state FROZEN and state TRAINING.
        rank_descriptor_frozen  : ModelDescriptor = None
        rank_descriptor_training: ModelDescriptor = None
        rank_descriptor_evaluate: ModelDescriptor = None
        for model_info in frozen_models:
            if model_info.model_name == ModelNames.AI_WORKLIST_RANK:
                rank_descriptor_frozen = model_info
                break

        for model_info in training_models:
            if model_info.model_name == ModelNames.AI_WORKLIST_RANK:
                rank_descriptor_training = model_info
                break

        if rank_descriptor_frozen is None:
            log.error(f'Failed to find model {ModelNames.AI_WORKLIST_RANK} for pipeline {AlgorithmNames.AI_WORKLIST} in state FROZEN. Performance evaluation cannot proceed.')
            return
        if rank_descriptor_training is None:
            log.error(f'Failed to find model {ModelNames.AI_WORKLIST_RANK} for pipeline {AlgorithmNames.AI_WORKLIST} in state TRAINING. Performance evaluation cannot proceed.')
            return

        # Create a copy of the latest TRAINING model in the EVALUATE state to avoid concurrent access to the model.
        rank_descriptor_evaluate = repo.create_model_version(rank_descriptor_training, repo.repository_root, ModelState.EVALUATE)
        if rank_descriptor_evaluate is None:
            log.error(f'Failed to create model {ModelNames.AI_WORKLIST_RANK} for pipeline {AlgorithmNames.AI_WORKLIST} in state EVALUATE. Performance evaluation cannot proceed.')
            return

        # At this point, we have the most recent model data, so reset the dirty flag.
        with self._performance_eval_lock as _:
            self._allow_perf_eval = False

        evaluate_pipeline_config: RankPipelineConfiguration = None
        frozen_pipeline_config  : RankPipelineConfiguration = None
        evaluate_resources      : RankModelResources        = None
        frozen_resources        : RankModelResources        = None
        evaluate_pipeline_model : RankStudyModel            = None
        frozen_pipeline_model   : RankStudyModel            = None
        model_compare           : ModelComparison           = None
        try: # Load resources for both the EVALUATE and FROZEN models into process memory.
            evaluate_resources  = RankModelResources(repo, rank_descriptor_evaluate)
            frozen_resources    = RankModelResources(repo, rank_descriptor_frozen)
            if not evaluate_resources.try_load_resources():
                log.error(f'Failed to load resources for model {rank_descriptor_evaluate.get_model_key()} version {rank_descriptor_evaluate.get_model_version()}; aborting performance comparison.')
                return
            if not frozen_resources.try_load_resources():
                log.error(f'Failed to load resources for model {rank_descriptor_frozen.get_model_key()} version {rank_descriptor_frozen.get_model_version()}; aborting performance comparison.')
                return

            # Create pipeline configurations for both models. Both are in read-only mode.
            evaluate_pipeline_config = RankPipelineConfiguration(evaluate_resources, train_model=False)
            evaluate_pipeline_model  = RankStudyModel(evaluate_pipeline_config)
            frozen_pipeline_config   = RankPipelineConfiguration(frozen_resources, train_model=False)
            frozen_pipeline_model    = RankStudyModel(frozen_pipeline_config)

            # Compare and retrieve the best model descriptor.
            log.info(f'Executing performance comparison for {AlgorithmNames.AI_WORKLIST} model {ModelNames.AI_WORKLIST_RANK}, {rank_descriptor_evaluate.get_model_key()} against {rank_descriptor_frozen.get_model_key()}.')
            model_compare = ModelComparison()
            best_model    = model_compare.compare(evaluate_pipeline_model, frozen_pipeline_model, baseline_data)
            if best_model is None:
                log.error(f'Failed to determine best performing model for {AlgorithmNames.AI_WORKLIST} model {ModelNames.AI_WORKLIST_RANK}.')
                return

            # If the model under evaluation performed the best, create a new model revision in both the FROZEN and TRAINING states.
            # Otherwise, create a new model revision based on the prior (frozen) model revision to "roll back" training done since that revision.
            # Create a new revision to force the primary node to reload the model weights and state.
            if best_model is evaluate_pipeline_model:
                new_revision: int = rank_descriptor_evaluate.model_revision + 1
                log.info(f'Creating new revision {new_revision} of {AlgorithmNames.AI_WORKLIST} model {ModelNames.AI_WORKLIST_RANK} based on the TRAINING state, since the updated model performed better.')
                _ = repo.create_model_version(rank_descriptor_training, repo.repository_root, ModelState.TRAINING, new_revision)
                _ = repo.create_model_version(rank_descriptor_frozen  , repo.repository_root, ModelState.FROZEN  , new_revision)
            else:
                new_revision: int = rank_descriptor_training.model_revision + 1
                log.info(f'Creating new revision {new_revision} of {AlgorithmNames.AI_WORKLIST} model {ModelNames.AI_WORKLIST_RANK} based on the FROZEN state, since the prior model performed better.')
                _ = repo.create_model_version(rank_descriptor_frozen, repo.repository_root, ModelState.TRAINING, new_revision)

            log.info(f'Pruning prior model revisions for {AlgorithmNames.AI_WORKLIST} model {ModelNames.AI_WORKLIST_RANK}.')
            repo.prune_unused_revisions(rank_descriptor_frozen  , ModelState.FROZEN  , keep_latest=2)
            repo.prune_unused_revisions(rank_descriptor_training, ModelState.TRAINING, keep_latest=1)

            log.info(f'Completed performance evaluation for {AlgorithmNames.AI_WORKLIST} pipeline.')

        except Exception as _: # pylint: disable=broad-except
            log.exception(f'Exception during {AlgorithmNames.AI_WORKLIST} model {ModelNames.AI_WORKLIST_RANK} performance evaluation')

        finally:
            if evaluate_resources is not None:
                evaluate_resources.unload_resources(do_gc=False)
            if frozen_resources is not None:
                frozen_resources.unload_resources(do_gc=False)

            repo.delete_model_revision(rank_descriptor_evaluate)
            evaluate_pipeline_config = None
            frozen_pipeline_config   = None
            evaluate_resources       = None
            frozen_resources         = None
            evaluate_pipeline_model  = None
            frozen_pipeline_model    = None
            model_compare            = None
            baseline_data            = None
            gc.collect()

    
    def get_external_user_id_mapping(self) -> Optional[UserIdMapping]:
        """
        Retrieve the mapping from external system user ID to internal pipeline user ID.

        Return
        ------
            An instance of `UserIdMapping`, or `None` if pipeline resources have not been loaded.
        """
        mapping: UserIdMapping = None
        with self._resource_loader_lock as _:
            if self._rank_model_resources is not None:
                mapping = self._rank_model_resources.user_mapping
        
        if mapping is None:
            log.error(f'No external user ID mapping defined for {AlgorithmNames.AI_WORKLIST} model {ModelNames.AI_WORKLIST_RANK}; are resources loaded?')
        
        return mapping


    def get_external_specialty_mapping(self) -> Optional[SpecialtyMapping]:
        """
        Retrieve the mapping from external system specialty ID to internal pipeline specialty ID.

        Return
        ------
            An instance of `SpecialtyMapping`, or `None` if pipeline resources have not been loaded.
        """
        mapping: SpecialtyMapping = None
        with self._resource_loader_lock as _:
            if self._rank_model_resources is not None:
                mapping = self._rank_model_resources.specialty_mapping
        
        if mapping is None:
            log.error(f'No SpecialtyMapping defined for {AlgorithmNames.AI_WORKLIST} model {ModelNames.AI_WORKLIST_RANK}; are resources loaded?')

        return mapping
    

    def get_performance_data(self) -> Optional[Tuple[AIWorklistPipelinePerformanceData, List[ModelDescriptor]]]:
        """
        Retrieve data that can be used to analyze the performance of the currently loaded model version.

        Returns
        -------
            A `tuple (AIWorklistPipelinePerformanceData, List[ModelDescriptor])` where:
            * The first element of the tuple contains data used to analyze model performance,
            * The second element of the tuple contains the list of `ModelDescriptor` identifying the current model(s) in use.
            If an error occurs, the returned tuple has values `(None, [])`.
        """
        models: List[ModelDescriptor]             = []
        stats : AIWorklistPipelinePerformanceData = None
        with self._resource_loader_lock as _:
            if self._rank_model_resources.resident_in_memory() and self._rank_model:
                rank   =  self._rank_model
                models = [self._rank_model_resources.model_descriptor]
                stats  = AIWorklistPipelinePerformanceData()
                stats.total_reward  = rank.total_reward
                stats.num_rewards   = rank.num_rewards
                stats.total_loss    = rank.agent.total_loss
                stats.num_loss      = rank.agent.num_loss
                stats.correct_count = rank.environment.correct_count
                stats.penalty_count = rank.environment.penalty_count

        return (stats, models)



class AIWorklistPipelineEvaluationThread(threading.Thread):
    """
    Implements a timer that performs periodic performance evaluations of the AI-worklist pipeline at a fixed time each day.

    Fields
    ------
    """
    def __init__(self, pipeline: AIWorklistPipeline, run_at_interval: timedelta, run_at_time: Optional[time]=None) -> None:
        super().__init__(name='AIPerfEval', daemon=True)

        if run_at_time is None:
            # By default, run at midnight local time.
            run_at_time = time(hour=0, minute=0, second=0)

        today    : date     = date.today()
        run_today: datetime = datetime(today.year, today.month, today.day, run_at_time.hour, run_at_time.minute, run_at_time.second)
        self.shutdown_signal: Event              = Event()
        self.run_at_interval: timedelta          = run_at_interval
        self.next_run_at    : datetime           = run_today
        self.pipeline       : AIWorklistPipeline = pipeline
        # Determine when the next run should start.
        while self.next_run_at.timestamp() < datetime.now().timestamp():
            self.next_run_at += run_at_interval


    def terminate(self) -> None:
        """
        Signal the thread to shut down immediately.
        """
        self.shutdown_signal.set()

    
    def run(self) -> None:
        """
        Run the main loop of the thread.
        """
        wait_seconds: float = (self.next_run_at - datetime.now()).total_seconds()
        if wait_seconds < 0:
            wait_seconds = 0
        log.info(f'The next automatic performance evaluation will be performed at {self.next_run_at.isoformat()}, in {wait_seconds} seconds.')

        while not self.shutdown_signal.wait(timeout=wait_seconds):
            try:
                # Schedule another run after the configured interval.
                self.next_run_at = self.next_run_at + self.run_at_interval
                # Execute the performance evaluation on this thread, since it won't be doing anything else.
                self.pipeline.evaluate_training_performance()
                # Wait until the next due time.
                wait_seconds = (self.next_run_at - datetime.now()).total_seconds()
                if wait_seconds < 0:
                    wait_seconds = 0
                log.info(f'The next automatic performance evaluation will be performed at {self.next_run_at.isoformat()}, in {wait_seconds} seconds.')
            except Exception as _: # pragma: no cover
                log.exception('An error occurred during an automated performance evaluation')
                wait_seconds = (self.next_run_at - datetime.now()).total_seconds()
                if wait_seconds < 0:
                    wait_seconds = 0
                log.info(f'The next automatic performance evaluation will be performed at {self.next_run_at.isoformat()}, in {wait_seconds} seconds.')

