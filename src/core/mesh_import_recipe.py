"""Closed, versioned execution contract for authoritative mesh imports.

The primary source digest identifies one file, while this recipe identifies how
the complete parser input becomes the source-space vertices and triangles used
by every later measurement. Self-contained imports retain strict v1; imports
that actually consume relative sidecars finalize as strict v2 with a closed
source manifest. Previously released five-field recipes and the official
two-field document fixture remain executable only through explicit legacy
profiles whose results are still checked by the geometry digest.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import re
from typing import Any, Mapping

from src.build_info import runtime_lock

from .source_manifest import (
    SOURCE_RESOLVER_PROFILE,
    SourceManifest,
    SourceManifestError,
)


MESH_IMPORT_RECIPE_ID = "org.archmeshrubbing.mesh-import.trimesh"
MESH_IMPORT_RECIPE_VERSION = "1.0.0"
MESH_IMPORT_RECIPE_VERSION_CLOSED_MANIFEST = "2.0.0"
MESH_IMPORT_LOADER = "trimesh"
MESH_IMPORT_FORCE = "mesh"
MESH_IMPORT_SANITIZER = "meshdata-v1"
MESH_IMPORT_SCENE_MERGE = "trimesh.util.concatenate/v1"
MESH_IMPORT_DEPENDENCY_POLICY = "deny_external"
MESH_IMPORT_CLOSED_MANIFEST_POLICY = "closed_manifest"

SUPPORTED_SOURCE_FORMATS = frozenset({"obj", "ply", "stl", "off", "gltf", "glb"})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STRICT_V1_KEYS = frozenset(
    {
        "recipe_id",
        "recipe_version",
        "format",
        "loader",
        "loader_version",
        "parser_runtime_sha256",
        "runtime_lock_sha256",
        "force",
        "process",
        "maintain_order",
        "scene_merge",
        "sanitizer",
        "dependency_policy",
    }
)
_STRICT_V2_KEYS = frozenset(
    {
        *_STRICT_V1_KEYS,
        "resolver_profile",
        "source_manifest",
    }
)
_LEGACY_FULL_KEYS = frozenset(
    {"format", "loader", "maintain_order", "process", "sanitizer"}
)
_LEGACY_DOCUMENT_KEYS = frozenset({"format", "process"})
_PARSER_RUNTIME_DISTRIBUTIONS = ("numpy", "pillow", "trimesh")

# How a recipe's recorded runtime identity is treated.
#
# `require_current` is for producing new authoritative geometry: a new
# GeometryRevision may only be decoded by the exact locked parser runtime.
#
# `record_only` is for reopening geometry that already exists.  The recorded
# version strings are validated for shape and kept as provenance, but they are
# not compared against the installed runtime, because equality of version
# strings was never the actual proof.  The proof is that the current parser
# re-decodes the same source bytes into the same geometry: `ArtifactSession`
# re-runs mesh admission and compares `accepted.geometry_sha256` against the
# GeometryRevision, then `_require_unchanged_source_geometry()` re-hashes the
# decoded mesh.  Gating on the version string ahead of that check only stopped
# the proof from being attempted, which made every upgrade of Trimesh orphan
# every project a researcher had already saved.
RUNTIME_POLICY_REQUIRE_CURRENT = "require_current"
RUNTIME_POLICY_RECORD_ONLY = "record_only"
SUPPORTED_RUNTIME_POLICIES = frozenset(
    {RUNTIME_POLICY_REQUIRE_CURRENT, RUNTIME_POLICY_RECORD_ONLY}
)


class MeshImportRecipeError(ValueError):
    """A mesh import recipe cannot be executed as the contract it claims."""


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MeshImportRecipeError(f"{field_name} must be a non-empty string")
    return value.strip()


def _source_format(value: object) -> str:
    source_format = _required_text(value, field_name="import_recipe.format")
    source_format = source_format.lower().removeprefix(".")
    if source_format not in SUPPORTED_SOURCE_FORMATS:
        raise MeshImportRecipeError(
            f"unsupported import_recipe.format: {source_format!r}"
        )
    return source_format


def _stored_source_format(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MeshImportRecipeError(
            "import_recipe.format must be a non-empty string"
        )
    normalized = value.strip().lower().removeprefix(".")
    if value != normalized:
        raise MeshImportRecipeError(
            "stored import_recipe.format must be normalized lowercase without "
            "whitespace or a leading dot"
        )
    return _source_format(value)


def _exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    profile: str,
) -> None:
    keys = set(value)
    if not all(isinstance(key, str) for key in keys):
        raise MeshImportRecipeError("import recipe field names must be strings")
    observed = {key for key in keys if isinstance(key, str)}
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing:
        raise MeshImportRecipeError(
            f"{profile} import recipe is missing fields: {', '.join(missing)}"
        )
    if unknown:
        raise MeshImportRecipeError(
            f"{profile} import recipe has unknown fields: {', '.join(unknown)}"
        )


def _current_runtime_identity(*, enforce_lock: bool = True) -> tuple[str, str, str]:
    """Return Trimesh version, parser-subset digest, and full lock digest.

    With `enforce_lock` the installed parser versions must equal the locked
    pins, which is the condition for producing new authoritative geometry.
    Without it the digest describes the versions that are actually installed,
    for recipes that record no runtime identity of their own and are only being
    reopened; a parser that is missing entirely is still an error, because
    nothing can be decoded without it.
    """

    try:
        _path, pins, lock_sha256 = runtime_lock()
    except RuntimeError as exc:
        raise MeshImportRecipeError(str(exc)) from exc
    parser_lines: list[str] = []
    observed_versions: dict[str, str] = {}
    for distribution in _PARSER_RUNTIME_DISTRIBUTIONS:
        locked = pins.get(distribution)
        if locked is None:
            raise MeshImportRecipeError(
                f"runtime lock has no exact {distribution} parser pin"
            )
        try:
            observed_version = importlib.metadata.version(locked[0])
        except importlib.metadata.PackageNotFoundError as exc:
            raise MeshImportRecipeError(
                f"required parser runtime is not installed: {locked[0]}"
            ) from exc
        if enforce_lock and observed_version != locked[1]:
            raise MeshImportRecipeError(
                f"installed {locked[0]} version does not match runtime lock: "
                f"{observed_version!r} != {locked[1]!r}"
            )
        observed_versions[distribution] = observed_version
        parser_lines.append(
            f"{distribution}=={locked[1] if enforce_lock else observed_version}\n"
        )
    if _SHA256_RE.fullmatch(lock_sha256) is None:
        raise MeshImportRecipeError("runtime lock SHA-256 is invalid")
    parser_runtime_sha256 = hashlib.sha256(
        "".join(parser_lines).encode("ascii")
    ).hexdigest()
    return observed_versions[MESH_IMPORT_LOADER], parser_runtime_sha256, lock_sha256


def _observed_loader_version() -> str | None:
    """Return the installed parser version without consulting the runtime lock.

    Reopening an existing document proves parser equivalence by re-decoding the
    source and reproducing ``GeometryRevision.geometry_sha256``, so the version
    string is evidence to display rather than a gate.  A missing distribution
    fails later, in the decode itself, with a message about the actual problem.
    """

    try:
        return importlib.metadata.version(MESH_IMPORT_LOADER)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True, slots=True)
class MeshImportExecution:
    """Validated parser execution values, including legacy classification."""

    source_format: str
    loader_version: str
    parser_runtime_sha256: str
    runtime_lock_sha256: str
    dependency_policy: str = MESH_IMPORT_DEPENDENCY_POLICY
    source_manifest: SourceManifest | None = None
    legacy_unversioned: bool = False
    reopened_under_runtime: str | None = None
    """Installed parser version when it differs from the recorded one.

    `None` means the runtime matches the record, or was not examined.  This is
    provenance for the verification report and the status line; it never
    changes what is accepted.
    """

    def strict_recipe(self) -> dict[str, Any]:
        if self.source_manifest is None:
            if self.dependency_policy != MESH_IMPORT_DEPENDENCY_POLICY:
                raise MeshImportRecipeError(
                    "a manifest-free execution must use dependency_policy=deny_external"
                )
            recipe_version = MESH_IMPORT_RECIPE_VERSION
        else:
            if self.dependency_policy != MESH_IMPORT_CLOSED_MANIFEST_POLICY:
                raise MeshImportRecipeError(
                    "a source manifest execution must use dependency_policy=closed_manifest"
                )
            recipe_version = MESH_IMPORT_RECIPE_VERSION_CLOSED_MANIFEST
        recipe: dict[str, Any] = {
            "recipe_id": MESH_IMPORT_RECIPE_ID,
            "recipe_version": recipe_version,
            "format": self.source_format,
            "loader": MESH_IMPORT_LOADER,
            "loader_version": self.loader_version,
            "parser_runtime_sha256": self.parser_runtime_sha256,
            "runtime_lock_sha256": self.runtime_lock_sha256,
            "force": MESH_IMPORT_FORCE,
            "process": False,
            "maintain_order": True,
            "scene_merge": MESH_IMPORT_SCENE_MERGE,
            "sanitizer": MESH_IMPORT_SANITIZER,
            "dependency_policy": self.dependency_policy,
        }
        if self.source_manifest is not None:
            recipe["resolver_profile"] = SOURCE_RESOLVER_PROFILE
            recipe["source_manifest"] = self.source_manifest.to_dict()
        return recipe


def current_mesh_import_recipe(source_format: str) -> dict[str, Any]:
    """Build the strict recipe actually supported by this runtime."""

    loader_version, parser_runtime_sha256, lock_sha256 = _current_runtime_identity()
    return MeshImportExecution(
        source_format=_source_format(source_format),
        loader_version=loader_version,
        parser_runtime_sha256=parser_runtime_sha256,
        runtime_lock_sha256=lock_sha256,
    ).strict_recipe()


def mesh_import_recipe_with_manifest(
    base_recipe: Mapping[str, object],
    manifest: SourceManifest,
) -> dict[str, Any]:
    """Finalize a new import receipt after recording its exact sidecar closure."""

    if not isinstance(manifest, SourceManifest):
        raise MeshImportRecipeError("manifest must be a SourceManifest")
    execution = validate_mesh_import_recipe(
        base_recipe,
        allow_legacy=False,
    )
    if execution.source_manifest is not None:
        raise MeshImportRecipeError("base recipe must not already contain a source manifest")
    if not manifest.dependency_entries:
        return dict(base_recipe)
    recipe = dict(base_recipe)
    recipe["recipe_version"] = MESH_IMPORT_RECIPE_VERSION_CLOSED_MANIFEST
    recipe["dependency_policy"] = MESH_IMPORT_CLOSED_MANIFEST_POLICY
    recipe["resolver_profile"] = SOURCE_RESOLVER_PROFILE
    recipe["source_manifest"] = manifest.to_dict()
    validate_mesh_import_recipe(recipe, allow_legacy=False)
    return recipe


def mesh_import_receipt_matches_base(
    base_recipe: Mapping[str, object],
    receipt: Mapping[str, object],
) -> bool:
    """Return whether a captured receipt executed the ticket's parser profile."""

    try:
        base = validate_mesh_import_recipe(base_recipe, allow_legacy=False)
        observed = validate_mesh_import_recipe(receipt, allow_legacy=False)
    except MeshImportRecipeError:
        return False
    if base.source_manifest is not None:
        return dict(base_recipe) == dict(receipt)
    shared_fields = _STRICT_V1_KEYS - {"recipe_version", "dependency_policy"}
    return all(base_recipe.get(key) == receipt.get(key) for key in shared_fields) and (
        base.source_format == observed.source_format
    )


def validate_mesh_import_recipe(
    value: Mapping[str, object],
    *,
    allow_legacy: bool,
    runtime_policy: str = RUNTIME_POLICY_REQUIRE_CURRENT,
) -> MeshImportExecution:
    """Validate one executable strict or explicitly supported legacy recipe.

    Under `require_current` a strict v1 or v2 recipe must match the installed
    parser version and the digest of the parser-relevant runtime subset, which
    is what producing new authoritative geometry requires. Under `record_only`
    those recorded values are validated for shape and kept as provenance, but
    not compared against the installed runtime, so that geometry which already
    exists stays readable; see `RUNTIME_POLICY_RECORD_ONLY`. The full lock
    digest is retained as build provenance but is never an execution gate under
    either policy: unrelated GUI/runtime pins must not make authoritative
    geometry unreadable. Legacy recipes can describe only the exact profiles
    emitted before v1; arbitrary mappings are never treated as executable
    instructions.
    """

    if runtime_policy not in SUPPORTED_RUNTIME_POLICIES:
        raise MeshImportRecipeError(f"unsupported runtime policy: {runtime_policy!r}")
    if not isinstance(value, Mapping):
        raise MeshImportRecipeError("import recipe must be a mapping")
    keys = set(value)
    is_legacy_full = keys == set(_LEGACY_FULL_KEYS)
    is_legacy_document = keys == set(_LEGACY_DOCUMENT_KEYS)
    is_legacy = is_legacy_full or is_legacy_document
    if is_legacy:
        if not allow_legacy:
            raise MeshImportRecipeError(
                "legacy unversioned import recipe is not allowed for new documents"
            )
        source_format = _stored_source_format(value.get("format"))
        if is_legacy_full and value.get("loader") != MESH_IMPORT_LOADER:
            raise MeshImportRecipeError("legacy import recipe loader must be trimesh")
        if value.get("process") is not False:
            raise MeshImportRecipeError("legacy import recipe process must be false")
        if is_legacy_full and value.get("maintain_order") is not True:
            raise MeshImportRecipeError(
                "legacy import recipe maintain_order must be true"
            )
        if is_legacy_full and value.get("sanitizer") != MESH_IMPORT_SANITIZER:
            raise MeshImportRecipeError(
                "legacy import recipe sanitizer must be meshdata-v1"
            )
        # A legacy recipe records no runtime identity at all, so there is
        # nothing to compare and the current one is filled in instead.  Under
        # `record_only` the runtime lock must not be consulted either: a
        # document saved before v1 would otherwise become unreadable the moment
        # any locked parser pin moved.
        loader_version, parser_runtime_sha256, lock_sha256 = _current_runtime_identity(
            enforce_lock=runtime_policy == RUNTIME_POLICY_REQUIRE_CURRENT,
        )
        return MeshImportExecution(
            source_format=source_format,
            loader_version=loader_version,
            parser_runtime_sha256=parser_runtime_sha256,
            runtime_lock_sha256=lock_sha256,
            legacy_unversioned=True,
        )

    recipe_version = value.get("recipe_version")
    if recipe_version == MESH_IMPORT_RECIPE_VERSION:
        _exact_keys(value, _STRICT_V1_KEYS, profile="strict v1")
        dependency_policy = MESH_IMPORT_DEPENDENCY_POLICY
        source_manifest = None
    elif recipe_version == MESH_IMPORT_RECIPE_VERSION_CLOSED_MANIFEST:
        _exact_keys(value, _STRICT_V2_KEYS, profile="strict v2")
        dependency_policy = MESH_IMPORT_CLOSED_MANIFEST_POLICY
        if value.get("resolver_profile") != SOURCE_RESOLVER_PROFILE:
            raise MeshImportRecipeError(
                "unsupported import_recipe.resolver_profile: "
                f"{value.get('resolver_profile')!r}"
            )
        raw_manifest = value.get("source_manifest")
        if not isinstance(raw_manifest, Mapping):
            raise MeshImportRecipeError("import_recipe.source_manifest must be an object")
        try:
            source_manifest = SourceManifest.from_dict(raw_manifest)
        except SourceManifestError as exc:
            raise MeshImportRecipeError(
                f"invalid import_recipe.source_manifest: {exc}"
            ) from exc
    else:
        raise MeshImportRecipeError(
            f"unsupported import_recipe.recipe_version: {recipe_version!r}"
        )
    source_format = _stored_source_format(value.get("format"))
    expected_scalars = {
        "recipe_id": MESH_IMPORT_RECIPE_ID,
        "loader": MESH_IMPORT_LOADER,
        "force": MESH_IMPORT_FORCE,
        "scene_merge": MESH_IMPORT_SCENE_MERGE,
        "sanitizer": MESH_IMPORT_SANITIZER,
        "dependency_policy": dependency_policy,
    }
    for field_name, expected in expected_scalars.items():
        if value.get(field_name) != expected:
            raise MeshImportRecipeError(
                f"unsupported import_recipe.{field_name}: {value.get(field_name)!r}"
            )
    if value.get("process") is not False:
        raise MeshImportRecipeError("import_recipe.process must be false")
    if value.get("maintain_order") is not True:
        raise MeshImportRecipeError("import_recipe.maintain_order must be true")

    loader_version = _required_text(
        value.get("loader_version"),
        field_name="import_recipe.loader_version",
    )
    if loader_version != value.get("loader_version"):
        raise MeshImportRecipeError(
            "stored import_recipe.loader_version must not contain surrounding whitespace"
        )
    parser_runtime_sha256 = _required_text(
        value.get("parser_runtime_sha256"),
        field_name="import_recipe.parser_runtime_sha256",
    )
    if parser_runtime_sha256 != value.get("parser_runtime_sha256"):
        raise MeshImportRecipeError(
            "stored import_recipe.parser_runtime_sha256 must not contain whitespace"
        )
    lock_sha256 = _required_text(
        value.get("runtime_lock_sha256"),
        field_name="import_recipe.runtime_lock_sha256",
    )
    if lock_sha256 != value.get("runtime_lock_sha256"):
        raise MeshImportRecipeError(
            "stored import_recipe.runtime_lock_sha256 must not contain whitespace"
        )
    if _SHA256_RE.fullmatch(parser_runtime_sha256) is None:
        raise MeshImportRecipeError(
            "import_recipe.parser_runtime_sha256 must be 64 lowercase hexadecimal characters"
        )
    if _SHA256_RE.fullmatch(lock_sha256) is None:
        raise MeshImportRecipeError(
            "import_recipe.runtime_lock_sha256 must be 64 lowercase hexadecimal characters"
        )

    reopened_under_runtime: str | None = None
    if runtime_policy == RUNTIME_POLICY_REQUIRE_CURRENT:
        current_version, current_parser_sha256, _current_lock_sha256 = (
            _current_runtime_identity()
        )
        if loader_version != current_version:
            raise MeshImportRecipeError(
                "saved Trimesh version does not match current runtime: "
                f"{loader_version!r} != {current_version!r}"
            )
        if parser_runtime_sha256 != current_parser_sha256:
            raise MeshImportRecipeError(
                "saved parser-runtime SHA-256 does not match current runtime"
            )
    else:
        observed_version = _observed_loader_version()
        if observed_version is not None and observed_version != loader_version:
            reopened_under_runtime = observed_version

    return MeshImportExecution(
        source_format=source_format,
        loader_version=loader_version,
        parser_runtime_sha256=parser_runtime_sha256,
        runtime_lock_sha256=lock_sha256,
        dependency_policy=dependency_policy,
        source_manifest=source_manifest,
        reopened_under_runtime=reopened_under_runtime,
    )


__all__ = [
    "MESH_IMPORT_FORCE",
    "MESH_IMPORT_DEPENDENCY_POLICY",
    "MESH_IMPORT_CLOSED_MANIFEST_POLICY",
    "MESH_IMPORT_LOADER",
    "MESH_IMPORT_RECIPE_ID",
    "MESH_IMPORT_RECIPE_VERSION",
    "MESH_IMPORT_RECIPE_VERSION_CLOSED_MANIFEST",
    "MESH_IMPORT_SANITIZER",
    "MESH_IMPORT_SCENE_MERGE",
    "MeshImportExecution",
    "MeshImportRecipeError",
    "RUNTIME_POLICY_RECORD_ONLY",
    "RUNTIME_POLICY_REQUIRE_CURRENT",
    "SUPPORTED_RUNTIME_POLICIES",
    "SUPPORTED_SOURCE_FORMATS",
    "current_mesh_import_recipe",
    "mesh_import_receipt_matches_base",
    "mesh_import_recipe_with_manifest",
    "validate_mesh_import_recipe",
]
