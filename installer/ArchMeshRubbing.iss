#ifndef AppVersion
  #error AppVersion must be supplied with /DAppVersion=<version>
#endif

#ifndef SourceDir
  #error SourceDir must be supplied with /DSourceDir=<absolute onedir path>
#endif

#ifndef OutputDir
  #error OutputDir must be supplied with /DOutputDir=<absolute output path>
#endif

#ifndef SourceCommit
  #error SourceCommit must be supplied with /DSourceCommit=<full commit hash>
#endif

#define AppName "ArchMeshRubbing"
#define AppExeName "ArchMeshRubbing.exe"

[Setup]
AppId={{B274D884-EE82-4B03-BB97-8EE62B89323A}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=ArchMeshRubbing contributors
AppPublisherURL=https://github.com/lzpxilfe/ArchMeshRubbing
AppSupportURL=https://github.com/lzpxilfe/ArchMeshRubbing/issues
AppComments=Unsigned verification build from source commit {#SourceCommit}
AppReadmeFile={app}\_internal\README.md
LicenseFile={#SourceDir}\_internal\LICENSE
DefaultDirName={localappdata}\Programs\ArchMeshRubbing
DefaultGroupName=ArchMeshRubbing
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#OutputDir}
OutputBaseFilename=ArchMeshRubbing-{#AppVersion}-Windows-x64-unsigned
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName={#AppName} {#AppVersion}
UninstallDisplayIcon={app}\{#AppExeName}
ChangesAssociations=no
CloseApplications=yes
RestartApplications=no
SetupLogging=no

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\ArchMeshRubbing\ArchMeshRubbing"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; Comment: "Offline cultural-heritage digital measurement workbench"
