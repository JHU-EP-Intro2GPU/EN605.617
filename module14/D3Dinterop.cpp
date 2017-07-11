<!DOCTYPE html>
<html class="" lang="en">
<head prefix="og: http://ogp.me/ns#">
<meta charset="utf-8">
<meta content="IE=edge" http-equiv="X-UA-Compatible">
<meta content="object" property="og:type">
<meta content="GitLab" property="og:site_name">
<meta content="opencl-book-samples/src/Chapter_11/D3Dinterop/D3Dinterop.cpp · master · EN-605.417.31_FA16 / intro_to_gpu" property="og:title">
<meta content="Base project for all code to be presented and worked on during the course" property="og:description">
<meta content="http://absaroka.jhuep.com/assets/gitlab_logo-7ae504fe4f68fdebb3c2034e36621930cd36ea87924c11ff65dbcb8ed50dca58.png" property="og:image">
<meta content="http://absaroka.jhuep.com/EN-605.417.31_FA16/intro_to_gpu/blob/master/opencl-book-samples/src/Chapter_11/D3Dinterop/D3Dinterop.cpp" property="og:url">
<meta content="summary" property="twitter:card">
<meta content="opencl-book-samples/src/Chapter_11/D3Dinterop/D3Dinterop.cpp · master · EN-605.417.31_FA16 / intro_to_gpu" property="twitter:title">
<meta content="Base project for all code to be presented and worked on during the course" property="twitter:description">
<meta content="http://absaroka.jhuep.com/assets/gitlab_logo-7ae504fe4f68fdebb3c2034e36621930cd36ea87924c11ff65dbcb8ed50dca58.png" property="twitter:image">

<title>opencl-book-samples/src/Chapter_11/D3Dinterop/D3Dinterop.cpp · master · EN-605.417.31_FA16 / intro_to_gpu · GitLab</title>
<meta content="Base project for all code to be presented and worked on during the course" name="description">
<link rel="shortcut icon" type="image/x-icon" href="/assets/favicon-075eba76312e8421991a0c1f89a89ee81678bcde72319dd3e8047e2a47cd3a42.ico" />
<link rel="stylesheet" media="all" href="/assets/application-b82c159e67a3d15c3f67bf6b7968181447bd0473e3acdf3b874759239ab1296b.css" />
<link rel="stylesheet" media="print" href="/assets/print-9c3a1eb4a2f45c9f3d7dd4de03f14c2e6b921e757168b595d7f161bbc320fc05.css" />
<script src="/assets/application-b6e6a0ec5d9fa435390d9f3cd075c95e666cffbe02f641b8b7cdcd9f3c168ed3.js"></script>
<meta name="csrf-param" content="authenticity_token" />
<meta name="csrf-token" content="dtXAbmz1x7SiCb2fmTnbyUI5CdcdXsCNeaBDUwlk7P065GPkJ/BWIwYMUXBeabb6WeDeBeeV46+Gy+LSMK2F6g==" />
<meta content="origin-when-cross-origin" name="referrer">
<meta content="width=device-width, initial-scale=1, maximum-scale=1" name="viewport">
<meta content="#474D57" name="theme-color">
<link rel="apple-touch-icon" type="image/x-icon" href="/assets/touch-icon-iphone-5a9cee0e8a51212e70b90c87c12f382c428870c0ff67d1eb034d884b78d2dae7.png" />
<link rel="apple-touch-icon" type="image/x-icon" href="/assets/touch-icon-ipad-a6eec6aeb9da138e507593b464fdac213047e49d3093fc30e90d9a995df83ba3.png" sizes="76x76" />
<link rel="apple-touch-icon" type="image/x-icon" href="/assets/touch-icon-iphone-retina-72e2aadf86513a56e050e7f0f2355deaa19cc17ed97bbe5147847f2748e5a3e3.png" sizes="120x120" />
<link rel="apple-touch-icon" type="image/x-icon" href="/assets/touch-icon-ipad-retina-8ebe416f5313483d9c1bc772b5bbe03ecad52a54eba443e5215a22caed2a16a2.png" sizes="152x152" />
<link color="rgb(226, 67, 41)" href="/assets/logo-d36b5212042cebc89b96df4bf6ac24e43db316143e89926c0db839ff694d2de4.svg" rel="mask-icon">
<meta content="/assets/msapplication-tile-1196ec67452f618d39cdd85e2e3a542f76574c071051ae7effbfde01710eb17d.png" name="msapplication-TileImage">
<meta content="#30353E" name="msapplication-TileColor">




</head>

<body class="ui_charcoal" data-group="" data-page="projects:blob:show" data-project="intro_to_gpu">
<script>
//<![CDATA[
window.gon={};gon.api_version="v3";gon.default_avatar_url="http:\/\/absaroka.jhuep.com\/assets\/no_avatar-849f9c04a3a0d0cea2424ae97b27447dc64a7dbfae83c036c45b403392f0e8ba.png";gon.max_file_size=10;gon.relative_url_root="";gon.shortcuts_path="\/help\/shortcuts";gon.user_color_scheme="white";gon.award_menu_url="\/emojis";gon.katex_css_url="\/assets\/katex-e46cafe9c3fa73920a7c2c063ee8bb0613e0cf85fd96a3aea25f8419c4bfcfba.css";gon.katex_js_url="\/assets\/katex-04bcf56379fcda0ee7c7a63f71d0fc15ffd2e014d017cd9d51fd6554dfccf40a.js";gon.current_user_id=5;
//]]>
</script>
<script>
  window.project_uploads_path = "/EN-605.417.31_FA16/intro_to_gpu/uploads";
  window.preview_markdown_path = "/EN-605.417.31_FA16/intro_to_gpu/preview_markdown";
</script>

<header class="navbar navbar-fixed-top navbar-gitlab with-horizontal-nav">
<a class="sr-only gl-accessibility" href="#content-body" tabindex="1">Skip to content</a>
<div class="container-fluid">
<div class="header-content">
<button aria-label="Toggle global navigation" class="side-nav-toggle" type="button">
<span class="sr-only">Toggle navigation</span>
<i class="fa fa-bars"></i>
</button>
<button class="navbar-toggle" type="button">
<span class="sr-only">Toggle navigation</span>
<i class="fa fa-ellipsis-v"></i>
</button>
<div class="navbar-collapse collapse">
<ul class="nav navbar-nav">
<li class="hidden-sm hidden-xs">
<div class="has-location-badge search search-form">
<form class="navbar-form" action="/search" accept-charset="UTF-8" method="get"><input name="utf8" type="hidden" value="&#x2713;" /><div class="search-input-container">
<div class="location-badge">This project</div>
<div class="search-input-wrap">
<div class="dropdown" data-url="/search/autocomplete">
<input type="search" name="search" id="search" placeholder="Search" class="search-input dropdown-menu-toggle no-outline js-search-dashboard-options" spellcheck="false" tabindex="1" autocomplete="off" data-toggle="dropdown" data-issues-path="http://absaroka.jhuep.com/dashboard/issues" data-mr-path="http://absaroka.jhuep.com/dashboard/merge_requests" />
<div class="dropdown-menu dropdown-select">
<div class="dropdown-content"><ul>
<li>
<a class="is-focused dropdown-menu-empty-link">
Loading...
</a>
</li>
</ul>
</div><div class="dropdown-loading"><i class="fa fa-spinner fa-spin"></i></div>
</div>
<i class="search-icon"></i>
<i class="clear-icon js-clear-input"></i>
</div>
</div>
</div>
<input type="hidden" name="group_id" id="group_id" class="js-search-group-options" />
<input type="hidden" name="project_id" id="search_project_id" value="68" class="js-search-project-options" data-project-path="intro_to_gpu" data-name="intro_to_gpu" data-issues-path="/EN-605.417.31_FA16/intro_to_gpu/issues" data-mr-path="/EN-605.417.31_FA16/intro_to_gpu/merge_requests" />
<input type="hidden" name="search_code" id="search_code" value="true" />
<input type="hidden" name="repository_ref" id="repository_ref" value="master" />

<div class="search-autocomplete-opts hide" data-autocomplete-path="/search/autocomplete" data-autocomplete-project-id="68" data-autocomplete-project-ref="master"></div>
</form></div>

</li>
<li class="visible-sm visible-xs">
<a title="Search" aria-label="Search" data-toggle="tooltip" data-placement="bottom" data-container="body" href="/search"><i class="fa fa-search"></i>
</a></li>
<li>
<a title="Admin Area" aria-label="Admin Area" data-toggle="tooltip" data-placement="bottom" data-container="body" href="/admin"><i class="fa fa-wrench fa-fw"></i>
</a></li>
<li>
<a title="Todos" aria-label="Todos" data-toggle="tooltip" data-placement="bottom" data-container="body" href="/dashboard/todos"><i class="fa fa-bell fa-fw"></i>
<span class="badge hidden todos-pending-count">
0
</span>
</a></li>
<li class="header-user dropdown">
<a class="header-user-dropdown-toggle" data-toggle="dropdown" href="/cpascal3"><img width="26" height="26" class="header-user-avatar" src="http://www.gravatar.com/avatar/6a7947cd3f7e4419520a1950de17b3be?s=52&amp;d=identicon" alt="6a7947cd3f7e4419520a1950de17b3be?s=52&amp;d=identicon" />
<i class="fa fa-caret-down"></i>
</a><div class="dropdown-menu-nav dropdown-menu-align-right">
<ul>
<li>
<a class="profile-link" aria-label="Profile" data-user="cpascal3" href="/cpascal3">Profile</a>
</li>
<li>
<a aria-label="Profile Settings" href="/profile">Profile Settings</a>
</li>
<li>
<a aria-label="Help" href="/help">Help</a>
</li>
<li class="divider"></li>
<li>
<a class="sign-out-link" aria-label="Sign out" rel="nofollow" data-method="delete" href="/users/sign_out">Sign out</a>
</li>
</ul>
</div>
</li>
</ul>
</div>
<h1 class="title"><span><a href="/EN-605.417.31_FA16">EN-605.417.31_FA16</a></span> / <a class="project-item-select-holder" href="/EN-605.417.31_FA16/intro_to_gpu">intro_to_gpu</a><button name="button" type="button" class="dropdown-toggle-caret js-projects-dropdown-toggle" aria-label="Toggle switch project dropdown" data-target=".js-dropdown-menu-projects" data-toggle="dropdown"><i class="fa fa-chevron-down"></i></button></h1>
<div class="header-logo">
<a class="home" title="Dashboard" id="logo" href="/"><svg width="36" height="36" class="tanuki-logo">
  <path class="tanuki-shape tanuki-left-ear" fill="#e24329" d="M2 14l9.38 9v-9l-4-12.28c-.205-.632-1.176-.632-1.38 0z"/>
  <path class="tanuki-shape tanuki-right-ear" fill="#e24329" d="M34 14l-9.38 9v-9l4-12.28c.205-.632 1.176-.632 1.38 0z"/>
  <path class="tanuki-shape tanuki-nose" fill="#e24329" d="M18,34.38 3,14 33,14 Z"/>
  <path class="tanuki-shape tanuki-left-eye" fill="#fc6d26" d="M18,34.38 11.38,14 2,14 6,25Z"/>
  <path class="tanuki-shape tanuki-right-eye" fill="#fc6d26" d="M18,34.38 24.62,14 34,14 30,25Z"/>
  <path class="tanuki-shape tanuki-left-cheek" fill="#fca326" d="M2 14L.1 20.16c-.18.565 0 1.2.5 1.56l17.42 12.66z"/>
  <path class="tanuki-shape tanuki-right-cheek" fill="#fca326" d="M34 14l1.9 6.16c.18.565 0 1.2-.5 1.56L18 34.38z"/>
</svg>

</a></div>
<div class="js-dropdown-menu-projects">
<div class="dropdown-menu dropdown-select dropdown-menu-projects">
<div class="dropdown-title"><span>Go to a project</span><button class="dropdown-title-button dropdown-menu-close" aria-label="Close" type="button"><i class="fa fa-times dropdown-menu-close-icon"></i></button></div>
<div class="dropdown-input"><input type="search" id="" class="dropdown-input-field" placeholder="Search your projects" autocomplete="off" /><i class="fa fa-search dropdown-input-search"></i><i role="button" class="fa fa-times dropdown-input-clear js-dropdown-input-clear"></i></div>
<div class="dropdown-content"></div>
<div class="dropdown-loading"><i class="fa fa-spinner fa-spin"></i></div>
</div>
</div>

</div>
</div>
</header>

<script>
  var findFileURL = "/EN-605.417.31_FA16/intro_to_gpu/find_file/master";
</script>

<div class="page-with-sidebar">
<div class="sidebar-wrapper nicescroll">
<div class="sidebar-action-buttons">
<div class="nav-header-btn toggle-nav-collapse" title="Open/Close">
<span class="sr-only">Toggle navigation</span>
<i class="fa fa-bars"></i>
</div>
<div class="nav-header-btn pin-nav-btn has-tooltip  js-nav-pin" data-container="body" data-placement="right" title="Pin Navigation">
<span class="sr-only">Toggle navigation pinning</span>
<i class="fa fa-fw fa-thumb-tack"></i>
</div>
</div>
<div class="nav-sidebar">
<ul class="nav">
<li class="active home"><a title="Projects" class="dashboard-shortcuts-projects" href="/dashboard/projects"><span>
Projects
</span>
</a></li><li class=""><a class="dashboard-shortcuts-activity" title="Activity" href="/dashboard/activity"><span>
Activity
</span>
</a></li><li class=""><a title="Groups" href="/dashboard/groups"><span>
Groups
</span>
</a></li><li class=""><a title="Milestones" href="/dashboard/milestones"><span>
Milestones
</span>
</a></li><li class=""><a title="Issues" class="dashboard-shortcuts-issues" href="/dashboard/issues?assignee_id=5"><span>
Issues
<span class="count">0</span>
</span>
</a></li><li class=""><a title="Merge Requests" class="dashboard-shortcuts-merge_requests" href="/dashboard/merge_requests?assignee_id=5"><span>
Merge Requests
<span class="count">0</span>
</span>
</a></li><li class=""><a title="Snippets" href="/dashboard/snippets"><span>
Snippets
</span>
</a></li></ul>
</div>

</div>
<div class="layout-nav">
<div class="container-fluid">
<div class="controls">
<div class="dropdown project-settings-dropdown">
<a class="dropdown-new btn btn-default" data-toggle="dropdown" href="#" id="project-settings-button">
<i class="fa fa-cog"></i>
<i class="fa fa-caret-down"></i>
</a>
<ul class="dropdown-menu dropdown-menu-align-right">
<li class=""><a title="Members" class="team-tab tab" href="/EN-605.417.31_FA16/intro_to_gpu/project_members"><span>
Members
</span>
</a></li><li class=""><a title="Groups" href="/EN-605.417.31_FA16/intro_to_gpu/group_links"><span>
Groups
</span>
</a></li><li class=""><a title="Deploy Keys" href="/EN-605.417.31_FA16/intro_to_gpu/deploy_keys"><span>
Deploy Keys
</span>
</a></li><li class=""><a title="Webhooks" href="/EN-605.417.31_FA16/intro_to_gpu/hooks"><span>
Webhooks
</span>
</a></li><li class=""><a title="Services" href="/EN-605.417.31_FA16/intro_to_gpu/services"><span>
Services
</span>
</a></li><li class=""><a title="Protected Branches" href="/EN-605.417.31_FA16/intro_to_gpu/protected_branches"><span>
Protected Branches
</span>
</a></li>
<li class="divider"></li>
<li>
<a href="/EN-605.417.31_FA16/intro_to_gpu/edit">Edit Project
</a></li>
</ul>
</div>
</div>
<div class="nav-control scrolling-tabs-container">
<div class="fade-left">
<i class="fa fa-angle-left"></i>
</div>
<div class="fade-right">
<i class="fa fa-angle-right"></i>
</div>
<ul class="nav-links scrolling-tabs">
<li class="home"><a title="Project" class="shortcuts-project" href="/EN-605.417.31_FA16/intro_to_gpu"><span>
Project
</span>
</a></li><li class=""><a title="Activity" class="shortcuts-project-activity" href="/EN-605.417.31_FA16/intro_to_gpu/activity"><span>
Activity
</span>
</a></li><li class="active"><a title="Repository" class="shortcuts-tree" href="/EN-605.417.31_FA16/intro_to_gpu/tree/master"><span>
Repository
</span>
</a></li><li class=""><a title="Container Registry" class="shortcuts-container-registry" href="/EN-605.417.31_FA16/intro_to_gpu/container_registry"><span>
Registry
</span>
</a></li><li class=""><a title="Graphs" class="shortcuts-graphs" href="/EN-605.417.31_FA16/intro_to_gpu/graphs/master"><span>
Graphs
</span>
</a></li><li class=""><a title="Issues" class="shortcuts-issues" href="/EN-605.417.31_FA16/intro_to_gpu/issues"><span>
Issues
<span class="badge count issue_counter">0</span>
</span>
</a></li><li class=""><a title="Merge Requests" class="shortcuts-merge_requests" href="/EN-605.417.31_FA16/intro_to_gpu/merge_requests"><span>
Merge Requests
<span class="badge count merge_counter">0</span>
</span>
</a></li><li class=""><a title="Wiki" class="shortcuts-wiki" href="/EN-605.417.31_FA16/intro_to_gpu/wikis/home"><span>
Wiki
</span>
</a></li><li class="hidden">
<a title="Network" class="shortcuts-network" href="/EN-605.417.31_FA16/intro_to_gpu/network/master">Network
</a></li>
<li class="hidden">
<a class="shortcuts-new-issue" href="/EN-605.417.31_FA16/intro_to_gpu/issues/new">Create a new issue
</a></li>
<li class="hidden">
<a title="Commits" class="shortcuts-commits" href="/EN-605.417.31_FA16/intro_to_gpu/commits/master">Commits
</a></li>
<li class="hidden">
<a title="Issue Boards" class="shortcuts-issue-boards" href="/EN-605.417.31_FA16/intro_to_gpu/boards">Issue Boards</a>
</li>
</ul>
</div>

</div>
</div>
<div class="content-wrapper page-with-layout-nav">
<div class="scrolling-tabs-container sub-nav-scroll">
<div class="fade-left">
<i class="fa fa-angle-left"></i>
</div>
<div class="fade-right">
<i class="fa fa-angle-right"></i>
</div>

<div class="nav-links sub-nav scrolling-tabs">
<ul class="container-fluid container-limited">
<li class="active"><a href="/EN-605.417.31_FA16/intro_to_gpu/tree/master">Files
</a></li><li class=""><a href="/EN-605.417.31_FA16/intro_to_gpu/commits/master">Commits
</a></li><li class=""><a href="/EN-605.417.31_FA16/intro_to_gpu/network/master">Network
</a></li><li class=""><a href="/EN-605.417.31_FA16/intro_to_gpu/compare?from=master&amp;to=master">Compare
</a></li><li class=""><a href="/EN-605.417.31_FA16/intro_to_gpu/branches">Branches
</a></li><li class=""><a href="/EN-605.417.31_FA16/intro_to_gpu/tags">Tags
</a></li></ul>
</div>
</div>

<div class="alert-wrapper">


<div class="flash-container flash-container-page">
</div>


</div>
<div class=" ">
<div class="content" id="content-body">

<div class="container-fluid container-limited">

<div class="tree-holder" id="tree-holder">
<div class="nav-block">
<div class="tree-ref-holder">
<form class="project-refs-form" action="/EN-605.417.31_FA16/intro_to_gpu/refs/switch" accept-charset="UTF-8" method="get"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="destination" id="destination" value="blob" />
<input type="hidden" name="path" id="path" value="opencl-book-samples/src/Chapter_11/D3Dinterop/D3Dinterop.cpp" />
<div class="dropdown">
<button class="dropdown-menu-toggle js-project-refs-dropdown" type="button" data-toggle="dropdown" data-selected="master" data-ref="master" data-refs-url="/EN-605.417.31_FA16/intro_to_gpu/refs" data-field-name="ref" data-submit-form-on-click="true"><span class="dropdown-toggle-text ">master</span><i class="fa fa-chevron-down"></i></button>
<div class="dropdown-menu dropdown-menu-selectable">
<div class="dropdown-title"><span>Switch branch/tag</span><button class="dropdown-title-button dropdown-menu-close" aria-label="Close" type="button"><i class="fa fa-times dropdown-menu-close-icon"></i></button></div>
<div class="dropdown-input"><input type="search" id="" class="dropdown-input-field" placeholder="Search branches and tags" autocomplete="off" /><i class="fa fa-search dropdown-input-search"></i><i role="button" class="fa fa-times dropdown-input-clear js-dropdown-input-clear"></i></div>
<div class="dropdown-content"></div>
<div class="dropdown-loading"><i class="fa fa-spinner fa-spin"></i></div>
</div>
</div>
</form>
</div>
<ul class="breadcrumb repo-breadcrumb">
<li>
<a href="/EN-605.417.31_FA16/intro_to_gpu/tree/master">intro_to_gpu
</a></li>
<li>
<a href="/EN-605.417.31_FA16/intro_to_gpu/tree/master/opencl-book-samples">opencl-book-samples</a>
</li>
<li>
<a href="/EN-605.417.31_FA16/intro_to_gpu/tree/master/opencl-book-samples/src">src</a>
</li>
<li>
<a href="/EN-605.417.31_FA16/intro_to_gpu/tree/master/opencl-book-samples/src/Chapter_11">Chapter_11</a>
</li>
<li>
<a href="/EN-605.417.31_FA16/intro_to_gpu/tree/master/opencl-book-samples/src/Chapter_11/D3Dinterop">D3Dinterop</a>
</li>
<li>
<a href="/EN-605.417.31_FA16/intro_to_gpu/blob/master/opencl-book-samples/src/Chapter_11/D3Dinterop/D3Dinterop.cpp"><strong>
D3Dinterop.cpp
</strong>
</a></li>
</ul>
</div>
<ul class="blob-commit-info hidden-xs">
<li class="commit js-toggle-container" id="commit-ca317aee">
<a href="/cpascal3"><img class="avatar has-tooltip s36 hidden-xs" alt="Chancellor Pascale&#39;s avatar" title="Chancellor Pascale" data-container="body" src="http://www.gravatar.com/avatar/6a7947cd3f7e4419520a1950de17b3be?s=72&amp;d=identicon" /></a>
<div class="commit-info-block">
<div class="commit-row-title">
<span class="item-title">
<a class="commit-row-message" href="/EN-605.417.31_FA16/intro_to_gpu/commit/ca317aee57f20e0a2303ed5d7da96d1c279991ac">committing the source code for the book right into the master branch, you might …</a>
<span class="commit-row-message visible-xs-inline">
&middot;
ca317aee
</span>
<a class="text-expander hidden-xs js-toggle-button">...</a>
</span>
<div class="commit-actions hidden-xs">
<button class="btn btn-clipboard btn-transparent" data-toggle="tooltip" data-placement="bottom" data-container="body" data-clipboard-text="ca317aee57f20e0a2303ed5d7da96d1c279991ac" type="button" title="Copy to clipboard"><i class="fa fa-clipboard"></i></button>
<a class="commit-short-id btn btn-transparent" href="/EN-605.417.31_FA16/intro_to_gpu/commit/ca317aee57f20e0a2303ed5d7da96d1c279991ac">ca317aee</a>

</div>
</div>
<pre class="commit-row-description js-toggle-content">…want to create duplicate folder for your specific system since using cmake creates a bunch of files and may be difficult to reconstruct this folder if stuff goes bad.</pre>
<a class="commit-author-link has-tooltip" title="cpascal3@jhu.edu" href="/cpascal3">Chancellor Pascale</a>
committed
<time class="js-timeago" title="Oct 11, 2015 5:53pm" datetime="2015-10-11T17:53:50Z" data-toggle="tooltip" data-placement="top" data-container="body">2015-10-11 13:53:50 -0400</time>
</div>
</li>

</ul>
<div class="blob-content-holder" id="blob-content-holder">
<article class="file-holder">
<div class="file-title">
<i class="fa fa-file-text-o fa-fw"></i>
<strong>
D3Dinterop.cpp
</strong>
<small>
24.4 KB
</small>
<div class="file-actions hidden-xs">
<div class="btn-group tree-btn-group">
<a class="btn btn-sm" target="_blank" href="/EN-605.417.31_FA16/intro_to_gpu/raw/master/opencl-book-samples/src/Chapter_11/D3Dinterop/D3Dinterop.cpp">Raw</a>
<a class="btn btn-sm" href="/EN-605.417.31_FA16/intro_to_gpu/blame/master/opencl-book-samples/src/Chapter_11/D3Dinterop/D3Dinterop.cpp">Blame</a>
<a class="btn btn-sm" href="/EN-605.417.31_FA16/intro_to_gpu/commits/master/opencl-book-samples/src/Chapter_11/D3Dinterop/D3Dinterop.cpp">History</a>
<a class="btn btn-sm" href="/EN-605.417.31_FA16/intro_to_gpu/blob/18ea4946eebf9c9b710405b0848cb85613d7ac47/opencl-book-samples/src/Chapter_11/D3Dinterop/D3Dinterop.cpp">Permalink</a>
</div>
<div class="btn-group" role="group">
<a class="btn btn-sm" href="/EN-605.417.31_FA16/intro_to_gpu/edit/master/opencl-book-samples/src/Chapter_11/D3Dinterop/D3Dinterop.cpp">Edit</a>
<button name="button" type="submit" class="btn btn-default" data-target="#modal-upload-blob" data-toggle="modal">Replace</button>
<button name="button" type="submit" class="btn btn-remove" data-target="#modal-remove-blob" data-toggle="modal">Delete</button>
</div>

</div>
</div>
<div class="file-content code js-syntax-highlight">
<div class="line-numbers">
<a class="diff-line-num" data-line-number="1" href="#L1" id="L1">
<i class="fa fa-link"></i>
1
</a>
<a class="diff-line-num" data-line-number="2" href="#L2" id="L2">
<i class="fa fa-link"></i>
2
</a>
<a class="diff-line-num" data-line-number="3" href="#L3" id="L3">
<i class="fa fa-link"></i>
3
</a>
<a class="diff-line-num" data-line-number="4" href="#L4" id="L4">
<i class="fa fa-link"></i>
4
</a>
<a class="diff-line-num" data-line-number="5" href="#L5" id="L5">
<i class="fa fa-link"></i>
5
</a>
<a class="diff-line-num" data-line-number="6" href="#L6" id="L6">
<i class="fa fa-link"></i>
6
</a>
<a class="diff-line-num" data-line-number="7" href="#L7" id="L7">
<i class="fa fa-link"></i>
7
</a>
<a class="diff-line-num" data-line-number="8" href="#L8" id="L8">
<i class="fa fa-link"></i>
8
</a>
<a class="diff-line-num" data-line-number="9" href="#L9" id="L9">
<i class="fa fa-link"></i>
9
</a>
<a class="diff-line-num" data-line-number="10" href="#L10" id="L10">
<i class="fa fa-link"></i>
10
</a>
<a class="diff-line-num" data-line-number="11" href="#L11" id="L11">
<i class="fa fa-link"></i>
11
</a>
<a class="diff-line-num" data-line-number="12" href="#L12" id="L12">
<i class="fa fa-link"></i>
12
</a>
<a class="diff-line-num" data-line-number="13" href="#L13" id="L13">
<i class="fa fa-link"></i>
13
</a>
<a class="diff-line-num" data-line-number="14" href="#L14" id="L14">
<i class="fa fa-link"></i>
14
</a>
<a class="diff-line-num" data-line-number="15" href="#L15" id="L15">
<i class="fa fa-link"></i>
15
</a>
<a class="diff-line-num" data-line-number="16" href="#L16" id="L16">
<i class="fa fa-link"></i>
16
</a>
<a class="diff-line-num" data-line-number="17" href="#L17" id="L17">
<i class="fa fa-link"></i>
17
</a>
<a class="diff-line-num" data-line-number="18" href="#L18" id="L18">
<i class="fa fa-link"></i>
18
</a>
<a class="diff-line-num" data-line-number="19" href="#L19" id="L19">
<i class="fa fa-link"></i>
19
</a>
<a class="diff-line-num" data-line-number="20" href="#L20" id="L20">
<i class="fa fa-link"></i>
20
</a>
<a class="diff-line-num" data-line-number="21" href="#L21" id="L21">
<i class="fa fa-link"></i>
21
</a>
<a class="diff-line-num" data-line-number="22" href="#L22" id="L22">
<i class="fa fa-link"></i>
22
</a>
<a class="diff-line-num" data-line-number="23" href="#L23" id="L23">
<i class="fa fa-link"></i>
23
</a>
<a class="diff-line-num" data-line-number="24" href="#L24" id="L24">
<i class="fa fa-link"></i>
24
</a>
<a class="diff-line-num" data-line-number="25" href="#L25" id="L25">
<i class="fa fa-link"></i>
25
</a>
<a class="diff-line-num" data-line-number="26" href="#L26" id="L26">
<i class="fa fa-link"></i>
26
</a>
<a class="diff-line-num" data-line-number="27" href="#L27" id="L27">
<i class="fa fa-link"></i>
27
</a>
<a class="diff-line-num" data-line-number="28" href="#L28" id="L28">
<i class="fa fa-link"></i>
28
</a>
<a class="diff-line-num" data-line-number="29" href="#L29" id="L29">
<i class="fa fa-link"></i>
29
</a>
<a class="diff-line-num" data-line-number="30" href="#L30" id="L30">
<i class="fa fa-link"></i>
30
</a>
<a class="diff-line-num" data-line-number="31" href="#L31" id="L31">
<i class="fa fa-link"></i>
31
</a>
<a class="diff-line-num" data-line-number="32" href="#L32" id="L32">
<i class="fa fa-link"></i>
32
</a>
<a class="diff-line-num" data-line-number="33" href="#L33" id="L33">
<i class="fa fa-link"></i>
33
</a>
<a class="diff-line-num" data-line-number="34" href="#L34" id="L34">
<i class="fa fa-link"></i>
34
</a>
<a class="diff-line-num" data-line-number="35" href="#L35" id="L35">
<i class="fa fa-link"></i>
35
</a>
<a class="diff-line-num" data-line-number="36" href="#L36" id="L36">
<i class="fa fa-link"></i>
36
</a>
<a class="diff-line-num" data-line-number="37" href="#L37" id="L37">
<i class="fa fa-link"></i>
37
</a>
<a class="diff-line-num" data-line-number="38" href="#L38" id="L38">
<i class="fa fa-link"></i>
38
</a>
<a class="diff-line-num" data-line-number="39" href="#L39" id="L39">
<i class="fa fa-link"></i>
39
</a>
<a class="diff-line-num" data-line-number="40" href="#L40" id="L40">
<i class="fa fa-link"></i>
40
</a>
<a class="diff-line-num" data-line-number="41" href="#L41" id="L41">
<i class="fa fa-link"></i>
41
</a>
<a class="diff-line-num" data-line-number="42" href="#L42" id="L42">
<i class="fa fa-link"></i>
42
</a>
<a class="diff-line-num" data-line-number="43" href="#L43" id="L43">
<i class="fa fa-link"></i>
43
</a>
<a class="diff-line-num" data-line-number="44" href="#L44" id="L44">
<i class="fa fa-link"></i>
44
</a>
<a class="diff-line-num" data-line-number="45" href="#L45" id="L45">
<i class="fa fa-link"></i>
45
</a>
<a class="diff-line-num" data-line-number="46" href="#L46" id="L46">
<i class="fa fa-link"></i>
46
</a>
<a class="diff-line-num" data-line-number="47" href="#L47" id="L47">
<i class="fa fa-link"></i>
47
</a>
<a class="diff-line-num" data-line-number="48" href="#L48" id="L48">
<i class="fa fa-link"></i>
48
</a>
<a class="diff-line-num" data-line-number="49" href="#L49" id="L49">
<i class="fa fa-link"></i>
49
</a>
<a class="diff-line-num" data-line-number="50" href="#L50" id="L50">
<i class="fa fa-link"></i>
50
</a>
<a class="diff-line-num" data-line-number="51" href="#L51" id="L51">
<i class="fa fa-link"></i>
51
</a>
<a class="diff-line-num" data-line-number="52" href="#L52" id="L52">
<i class="fa fa-link"></i>
52
</a>
<a class="diff-line-num" data-line-number="53" href="#L53" id="L53">
<i class="fa fa-link"></i>
53
</a>
<a class="diff-line-num" data-line-number="54" href="#L54" id="L54">
<i class="fa fa-link"></i>
54
</a>
<a class="diff-line-num" data-line-number="55" href="#L55" id="L55">
<i class="fa fa-link"></i>
55
</a>
<a class="diff-line-num" data-line-number="56" href="#L56" id="L56">
<i class="fa fa-link"></i>
56
</a>
<a class="diff-line-num" data-line-number="57" href="#L57" id="L57">
<i class="fa fa-link"></i>
57
</a>
<a class="diff-line-num" data-line-number="58" href="#L58" id="L58">
<i class="fa fa-link"></i>
58
</a>
<a class="diff-line-num" data-line-number="59" href="#L59" id="L59">
<i class="fa fa-link"></i>
59
</a>
<a class="diff-line-num" data-line-number="60" href="#L60" id="L60">
<i class="fa fa-link"></i>
60
</a>
<a class="diff-line-num" data-line-number="61" href="#L61" id="L61">
<i class="fa fa-link"></i>
61
</a>
<a class="diff-line-num" data-line-number="62" href="#L62" id="L62">
<i class="fa fa-link"></i>
62
</a>
<a class="diff-line-num" data-line-number="63" href="#L63" id="L63">
<i class="fa fa-link"></i>
63
</a>
<a class="diff-line-num" data-line-number="64" href="#L64" id="L64">
<i class="fa fa-link"></i>
64
</a>
<a class="diff-line-num" data-line-number="65" href="#L65" id="L65">
<i class="fa fa-link"></i>
65
</a>
<a class="diff-line-num" data-line-number="66" href="#L66" id="L66">
<i class="fa fa-link"></i>
66
</a>
<a class="diff-line-num" data-line-number="67" href="#L67" id="L67">
<i class="fa fa-link"></i>
67
</a>
<a class="diff-line-num" data-line-number="68" href="#L68" id="L68">
<i class="fa fa-link"></i>
68
</a>
<a class="diff-line-num" data-line-number="69" href="#L69" id="L69">
<i class="fa fa-link"></i>
69
</a>
<a class="diff-line-num" data-line-number="70" href="#L70" id="L70">
<i class="fa fa-link"></i>
70
</a>
<a class="diff-line-num" data-line-number="71" href="#L71" id="L71">
<i class="fa fa-link"></i>
71
</a>
<a class="diff-line-num" data-line-number="72" href="#L72" id="L72">
<i class="fa fa-link"></i>
72
</a>
<a class="diff-line-num" data-line-number="73" href="#L73" id="L73">
<i class="fa fa-link"></i>
73
</a>
<a class="diff-line-num" data-line-number="74" href="#L74" id="L74">
<i class="fa fa-link"></i>
74
</a>
<a class="diff-line-num" data-line-number="75" href="#L75" id="L75">
<i class="fa fa-link"></i>
75
</a>
<a class="diff-line-num" data-line-number="76" href="#L76" id="L76">
<i class="fa fa-link"></i>
76
</a>
<a class="diff-line-num" data-line-number="77" href="#L77" id="L77">
<i class="fa fa-link"></i>
77
</a>
<a class="diff-line-num" data-line-number="78" href="#L78" id="L78">
<i class="fa fa-link"></i>
78
</a>
<a class="diff-line-num" data-line-number="79" href="#L79" id="L79">
<i class="fa fa-link"></i>
79
</a>
<a class="diff-line-num" data-line-number="80" href="#L80" id="L80">
<i class="fa fa-link"></i>
80
</a>
<a class="diff-line-num" data-line-number="81" href="#L81" id="L81">
<i class="fa fa-link"></i>
81
</a>
<a class="diff-line-num" data-line-number="82" href="#L82" id="L82">
<i class="fa fa-link"></i>
82
</a>
<a class="diff-line-num" data-line-number="83" href="#L83" id="L83">
<i class="fa fa-link"></i>
83
</a>
<a class="diff-line-num" data-line-number="84" href="#L84" id="L84">
<i class="fa fa-link"></i>
84
</a>
<a class="diff-line-num" data-line-number="85" href="#L85" id="L85">
<i class="fa fa-link"></i>
85
</a>
<a class="diff-line-num" data-line-number="86" href="#L86" id="L86">
<i class="fa fa-link"></i>
86
</a>
<a class="diff-line-num" data-line-number="87" href="#L87" id="L87">
<i class="fa fa-link"></i>
87
</a>
<a class="diff-line-num" data-line-number="88" href="#L88" id="L88">
<i class="fa fa-link"></i>
88
</a>
<a class="diff-line-num" data-line-number="89" href="#L89" id="L89">
<i class="fa fa-link"></i>
89
</a>
<a class="diff-line-num" data-line-number="90" href="#L90" id="L90">
<i class="fa fa-link"></i>
90
</a>
<a class="diff-line-num" data-line-number="91" href="#L91" id="L91">
<i class="fa fa-link"></i>
91
</a>
<a class="diff-line-num" data-line-number="92" href="#L92" id="L92">
<i class="fa fa-link"></i>
92
</a>
<a class="diff-line-num" data-line-number="93" href="#L93" id="L93">
<i class="fa fa-link"></i>
93
</a>
<a class="diff-line-num" data-line-number="94" href="#L94" id="L94">
<i class="fa fa-link"></i>
94
</a>
<a class="diff-line-num" data-line-number="95" href="#L95" id="L95">
<i class="fa fa-link"></i>
95
</a>
<a class="diff-line-num" data-line-number="96" href="#L96" id="L96">
<i class="fa fa-link"></i>
96
</a>
<a class="diff-line-num" data-line-number="97" href="#L97" id="L97">
<i class="fa fa-link"></i>
97
</a>
<a class="diff-line-num" data-line-number="98" href="#L98" id="L98">
<i class="fa fa-link"></i>
98
</a>
<a class="diff-line-num" data-line-number="99" href="#L99" id="L99">
<i class="fa fa-link"></i>
99
</a>
<a class="diff-line-num" data-line-number="100" href="#L100" id="L100">
<i class="fa fa-link"></i>
100
</a>
<a class="diff-line-num" data-line-number="101" href="#L101" id="L101">
<i class="fa fa-link"></i>
101
</a>
<a class="diff-line-num" data-line-number="102" href="#L102" id="L102">
<i class="fa fa-link"></i>
102
</a>
<a class="diff-line-num" data-line-number="103" href="#L103" id="L103">
<i class="fa fa-link"></i>
103
</a>
<a class="diff-line-num" data-line-number="104" href="#L104" id="L104">
<i class="fa fa-link"></i>
104
</a>
<a class="diff-line-num" data-line-number="105" href="#L105" id="L105">
<i class="fa fa-link"></i>
105
</a>
<a class="diff-line-num" data-line-number="106" href="#L106" id="L106">
<i class="fa fa-link"></i>
106
</a>
<a class="diff-line-num" data-line-number="107" href="#L107" id="L107">
<i class="fa fa-link"></i>
107
</a>
<a class="diff-line-num" data-line-number="108" href="#L108" id="L108">
<i class="fa fa-link"></i>
108
</a>
<a class="diff-line-num" data-line-number="109" href="#L109" id="L109">
<i class="fa fa-link"></i>
109
</a>
<a class="diff-line-num" data-line-number="110" href="#L110" id="L110">
<i class="fa fa-link"></i>
110
</a>
<a class="diff-line-num" data-line-number="111" href="#L111" id="L111">
<i class="fa fa-link"></i>
111
</a>
<a class="diff-line-num" data-line-number="112" href="#L112" id="L112">
<i class="fa fa-link"></i>
112
</a>
<a class="diff-line-num" data-line-number="113" href="#L113" id="L113">
<i class="fa fa-link"></i>
113
</a>
<a class="diff-line-num" data-line-number="114" href="#L114" id="L114">
<i class="fa fa-link"></i>
114
</a>
<a class="diff-line-num" data-line-number="115" href="#L115" id="L115">
<i class="fa fa-link"></i>
115
</a>
<a class="diff-line-num" data-line-number="116" href="#L116" id="L116">
<i class="fa fa-link"></i>
116
</a>
<a class="diff-line-num" data-line-number="117" href="#L117" id="L117">
<i class="fa fa-link"></i>
117
</a>
<a class="diff-line-num" data-line-number="118" href="#L118" id="L118">
<i class="fa fa-link"></i>
118
</a>
<a class="diff-line-num" data-line-number="119" href="#L119" id="L119">
<i class="fa fa-link"></i>
119
</a>
<a class="diff-line-num" data-line-number="120" href="#L120" id="L120">
<i class="fa fa-link"></i>
120
</a>
<a class="diff-line-num" data-line-number="121" href="#L121" id="L121">
<i class="fa fa-link"></i>
121
</a>
<a class="diff-line-num" data-line-number="122" href="#L122" id="L122">
<i class="fa fa-link"></i>
122
</a>
<a class="diff-line-num" data-line-number="123" href="#L123" id="L123">
<i class="fa fa-link"></i>
123
</a>
<a class="diff-line-num" data-line-number="124" href="#L124" id="L124">
<i class="fa fa-link"></i>
124
</a>
<a class="diff-line-num" data-line-number="125" href="#L125" id="L125">
<i class="fa fa-link"></i>
125
</a>
<a class="diff-line-num" data-line-number="126" href="#L126" id="L126">
<i class="fa fa-link"></i>
126
</a>
<a class="diff-line-num" data-line-number="127" href="#L127" id="L127">
<i class="fa fa-link"></i>
127
</a>
<a class="diff-line-num" data-line-number="128" href="#L128" id="L128">
<i class="fa fa-link"></i>
128
</a>
<a class="diff-line-num" data-line-number="129" href="#L129" id="L129">
<i class="fa fa-link"></i>
129
</a>
<a class="diff-line-num" data-line-number="130" href="#L130" id="L130">
<i class="fa fa-link"></i>
130
</a>
<a class="diff-line-num" data-line-number="131" href="#L131" id="L131">
<i class="fa fa-link"></i>
131
</a>
<a class="diff-line-num" data-line-number="132" href="#L132" id="L132">
<i class="fa fa-link"></i>
132
</a>
<a class="diff-line-num" data-line-number="133" href="#L133" id="L133">
<i class="fa fa-link"></i>
133
</a>
<a class="diff-line-num" data-line-number="134" href="#L134" id="L134">
<i class="fa fa-link"></i>
134
</a>
<a class="diff-line-num" data-line-number="135" href="#L135" id="L135">
<i class="fa fa-link"></i>
135
</a>
<a class="diff-line-num" data-line-number="136" href="#L136" id="L136">
<i class="fa fa-link"></i>
136
</a>
<a class="diff-line-num" data-line-number="137" href="#L137" id="L137">
<i class="fa fa-link"></i>
137
</a>
<a class="diff-line-num" data-line-number="138" href="#L138" id="L138">
<i class="fa fa-link"></i>
138
</a>
<a class="diff-line-num" data-line-number="139" href="#L139" id="L139">
<i class="fa fa-link"></i>
139
</a>
<a class="diff-line-num" data-line-number="140" href="#L140" id="L140">
<i class="fa fa-link"></i>
140
</a>
<a class="diff-line-num" data-line-number="141" href="#L141" id="L141">
<i class="fa fa-link"></i>
141
</a>
<a class="diff-line-num" data-line-number="142" href="#L142" id="L142">
<i class="fa fa-link"></i>
142
</a>
<a class="diff-line-num" data-line-number="143" href="#L143" id="L143">
<i class="fa fa-link"></i>
143
</a>
<a class="diff-line-num" data-line-number="144" href="#L144" id="L144">
<i class="fa fa-link"></i>
144
</a>
<a class="diff-line-num" data-line-number="145" href="#L145" id="L145">
<i class="fa fa-link"></i>
145
</a>
<a class="diff-line-num" data-line-number="146" href="#L146" id="L146">
<i class="fa fa-link"></i>
146
</a>
<a class="diff-line-num" data-line-number="147" href="#L147" id="L147">
<i class="fa fa-link"></i>
147
</a>
<a class="diff-line-num" data-line-number="148" href="#L148" id="L148">
<i class="fa fa-link"></i>
148
</a>
<a class="diff-line-num" data-line-number="149" href="#L149" id="L149">
<i class="fa fa-link"></i>
149
</a>
<a class="diff-line-num" data-line-number="150" href="#L150" id="L150">
<i class="fa fa-link"></i>
150
</a>
<a class="diff-line-num" data-line-number="151" href="#L151" id="L151">
<i class="fa fa-link"></i>
151
</a>
<a class="diff-line-num" data-line-number="152" href="#L152" id="L152">
<i class="fa fa-link"></i>
152
</a>
<a class="diff-line-num" data-line-number="153" href="#L153" id="L153">
<i class="fa fa-link"></i>
153
</a>
<a class="diff-line-num" data-line-number="154" href="#L154" id="L154">
<i class="fa fa-link"></i>
154
</a>
<a class="diff-line-num" data-line-number="155" href="#L155" id="L155">
<i class="fa fa-link"></i>
155
</a>
<a class="diff-line-num" data-line-number="156" href="#L156" id="L156">
<i class="fa fa-link"></i>
156
</a>
<a class="diff-line-num" data-line-number="157" href="#L157" id="L157">
<i class="fa fa-link"></i>
157
</a>
<a class="diff-line-num" data-line-number="158" href="#L158" id="L158">
<i class="fa fa-link"></i>
158
</a>
<a class="diff-line-num" data-line-number="159" href="#L159" id="L159">
<i class="fa fa-link"></i>
159
</a>
<a class="diff-line-num" data-line-number="160" href="#L160" id="L160">
<i class="fa fa-link"></i>
160
</a>
<a class="diff-line-num" data-line-number="161" href="#L161" id="L161">
<i class="fa fa-link"></i>
161
</a>
<a class="diff-line-num" data-line-number="162" href="#L162" id="L162">
<i class="fa fa-link"></i>
162
</a>
<a class="diff-line-num" data-line-number="163" href="#L163" id="L163">
<i class="fa fa-link"></i>
163
</a>
<a class="diff-line-num" data-line-number="164" href="#L164" id="L164">
<i class="fa fa-link"></i>
164
</a>
<a class="diff-line-num" data-line-number="165" href="#L165" id="L165">
<i class="fa fa-link"></i>
165
</a>
<a class="diff-line-num" data-line-number="166" href="#L166" id="L166">
<i class="fa fa-link"></i>
166
</a>
<a class="diff-line-num" data-line-number="167" href="#L167" id="L167">
<i class="fa fa-link"></i>
167
</a>
<a class="diff-line-num" data-line-number="168" href="#L168" id="L168">
<i class="fa fa-link"></i>
168
</a>
<a class="diff-line-num" data-line-number="169" href="#L169" id="L169">
<i class="fa fa-link"></i>
169
</a>
<a class="diff-line-num" data-line-number="170" href="#L170" id="L170">
<i class="fa fa-link"></i>
170
</a>
<a class="diff-line-num" data-line-number="171" href="#L171" id="L171">
<i class="fa fa-link"></i>
171
</a>
<a class="diff-line-num" data-line-number="172" href="#L172" id="L172">
<i class="fa fa-link"></i>
172
</a>
<a class="diff-line-num" data-line-number="173" href="#L173" id="L173">
<i class="fa fa-link"></i>
173
</a>
<a class="diff-line-num" data-line-number="174" href="#L174" id="L174">
<i class="fa fa-link"></i>
174
</a>
<a class="diff-line-num" data-line-number="175" href="#L175" id="L175">
<i class="fa fa-link"></i>
175
</a>
<a class="diff-line-num" data-line-number="176" href="#L176" id="L176">
<i class="fa fa-link"></i>
176
</a>
<a class="diff-line-num" data-line-number="177" href="#L177" id="L177">
<i class="fa fa-link"></i>
177
</a>
<a class="diff-line-num" data-line-number="178" href="#L178" id="L178">
<i class="fa fa-link"></i>
178
</a>
<a class="diff-line-num" data-line-number="179" href="#L179" id="L179">
<i class="fa fa-link"></i>
179
</a>
<a class="diff-line-num" data-line-number="180" href="#L180" id="L180">
<i class="fa fa-link"></i>
180
</a>
<a class="diff-line-num" data-line-number="181" href="#L181" id="L181">
<i class="fa fa-link"></i>
181
</a>
<a class="diff-line-num" data-line-number="182" href="#L182" id="L182">
<i class="fa fa-link"></i>
182
</a>
<a class="diff-line-num" data-line-number="183" href="#L183" id="L183">
<i class="fa fa-link"></i>
183
</a>
<a class="diff-line-num" data-line-number="184" href="#L184" id="L184">
<i class="fa fa-link"></i>
184
</a>
<a class="diff-line-num" data-line-number="185" href="#L185" id="L185">
<i class="fa fa-link"></i>
185
</a>
<a class="diff-line-num" data-line-number="186" href="#L186" id="L186">
<i class="fa fa-link"></i>
186
</a>
<a class="diff-line-num" data-line-number="187" href="#L187" id="L187">
<i class="fa fa-link"></i>
187
</a>
<a class="diff-line-num" data-line-number="188" href="#L188" id="L188">
<i class="fa fa-link"></i>
188
</a>
<a class="diff-line-num" data-line-number="189" href="#L189" id="L189">
<i class="fa fa-link"></i>
189
</a>
<a class="diff-line-num" data-line-number="190" href="#L190" id="L190">
<i class="fa fa-link"></i>
190
</a>
<a class="diff-line-num" data-line-number="191" href="#L191" id="L191">
<i class="fa fa-link"></i>
191
</a>
<a class="diff-line-num" data-line-number="192" href="#L192" id="L192">
<i class="fa fa-link"></i>
192
</a>
<a class="diff-line-num" data-line-number="193" href="#L193" id="L193">
<i class="fa fa-link"></i>
193
</a>
<a class="diff-line-num" data-line-number="194" href="#L194" id="L194">
<i class="fa fa-link"></i>
194
</a>
<a class="diff-line-num" data-line-number="195" href="#L195" id="L195">
<i class="fa fa-link"></i>
195
</a>
<a class="diff-line-num" data-line-number="196" href="#L196" id="L196">
<i class="fa fa-link"></i>
196
</a>
<a class="diff-line-num" data-line-number="197" href="#L197" id="L197">
<i class="fa fa-link"></i>
197
</a>
<a class="diff-line-num" data-line-number="198" href="#L198" id="L198">
<i class="fa fa-link"></i>
198
</a>
<a class="diff-line-num" data-line-number="199" href="#L199" id="L199">
<i class="fa fa-link"></i>
199
</a>
<a class="diff-line-num" data-line-number="200" href="#L200" id="L200">
<i class="fa fa-link"></i>
200
</a>
<a class="diff-line-num" data-line-number="201" href="#L201" id="L201">
<i class="fa fa-link"></i>
201
</a>
<a class="diff-line-num" data-line-number="202" href="#L202" id="L202">
<i class="fa fa-link"></i>
202
</a>
<a class="diff-line-num" data-line-number="203" href="#L203" id="L203">
<i class="fa fa-link"></i>
203
</a>
<a class="diff-line-num" data-line-number="204" href="#L204" id="L204">
<i class="fa fa-link"></i>
204
</a>
<a class="diff-line-num" data-line-number="205" href="#L205" id="L205">
<i class="fa fa-link"></i>
205
</a>
<a class="diff-line-num" data-line-number="206" href="#L206" id="L206">
<i class="fa fa-link"></i>
206
</a>
<a class="diff-line-num" data-line-number="207" href="#L207" id="L207">
<i class="fa fa-link"></i>
207
</a>
<a class="diff-line-num" data-line-number="208" href="#L208" id="L208">
<i class="fa fa-link"></i>
208
</a>
<a class="diff-line-num" data-line-number="209" href="#L209" id="L209">
<i class="fa fa-link"></i>
209
</a>
<a class="diff-line-num" data-line-number="210" href="#L210" id="L210">
<i class="fa fa-link"></i>
210
</a>
<a class="diff-line-num" data-line-number="211" href="#L211" id="L211">
<i class="fa fa-link"></i>
211
</a>
<a class="diff-line-num" data-line-number="212" href="#L212" id="L212">
<i class="fa fa-link"></i>
212
</a>
<a class="diff-line-num" data-line-number="213" href="#L213" id="L213">
<i class="fa fa-link"></i>
213
</a>
<a class="diff-line-num" data-line-number="214" href="#L214" id="L214">
<i class="fa fa-link"></i>
214
</a>
<a class="diff-line-num" data-line-number="215" href="#L215" id="L215">
<i class="fa fa-link"></i>
215
</a>
<a class="diff-line-num" data-line-number="216" href="#L216" id="L216">
<i class="fa fa-link"></i>
216
</a>
<a class="diff-line-num" data-line-number="217" href="#L217" id="L217">
<i class="fa fa-link"></i>
217
</a>
<a class="diff-line-num" data-line-number="218" href="#L218" id="L218">
<i class="fa fa-link"></i>
218
</a>
<a class="diff-line-num" data-line-number="219" href="#L219" id="L219">
<i class="fa fa-link"></i>
219
</a>
<a class="diff-line-num" data-line-number="220" href="#L220" id="L220">
<i class="fa fa-link"></i>
220
</a>
<a class="diff-line-num" data-line-number="221" href="#L221" id="L221">
<i class="fa fa-link"></i>
221
</a>
<a class="diff-line-num" data-line-number="222" href="#L222" id="L222">
<i class="fa fa-link"></i>
222
</a>
<a class="diff-line-num" data-line-number="223" href="#L223" id="L223">
<i class="fa fa-link"></i>
223
</a>
<a class="diff-line-num" data-line-number="224" href="#L224" id="L224">
<i class="fa fa-link"></i>
224
</a>
<a class="diff-line-num" data-line-number="225" href="#L225" id="L225">
<i class="fa fa-link"></i>
225
</a>
<a class="diff-line-num" data-line-number="226" href="#L226" id="L226">
<i class="fa fa-link"></i>
226
</a>
<a class="diff-line-num" data-line-number="227" href="#L227" id="L227">
<i class="fa fa-link"></i>
227
</a>
<a class="diff-line-num" data-line-number="228" href="#L228" id="L228">
<i class="fa fa-link"></i>
228
</a>
<a class="diff-line-num" data-line-number="229" href="#L229" id="L229">
<i class="fa fa-link"></i>
229
</a>
<a class="diff-line-num" data-line-number="230" href="#L230" id="L230">
<i class="fa fa-link"></i>
230
</a>
<a class="diff-line-num" data-line-number="231" href="#L231" id="L231">
<i class="fa fa-link"></i>
231
</a>
<a class="diff-line-num" data-line-number="232" href="#L232" id="L232">
<i class="fa fa-link"></i>
232
</a>
<a class="diff-line-num" data-line-number="233" href="#L233" id="L233">
<i class="fa fa-link"></i>
233
</a>
<a class="diff-line-num" data-line-number="234" href="#L234" id="L234">
<i class="fa fa-link"></i>
234
</a>
<a class="diff-line-num" data-line-number="235" href="#L235" id="L235">
<i class="fa fa-link"></i>
235
</a>
<a class="diff-line-num" data-line-number="236" href="#L236" id="L236">
<i class="fa fa-link"></i>
236
</a>
<a class="diff-line-num" data-line-number="237" href="#L237" id="L237">
<i class="fa fa-link"></i>
237
</a>
<a class="diff-line-num" data-line-number="238" href="#L238" id="L238">
<i class="fa fa-link"></i>
238
</a>
<a class="diff-line-num" data-line-number="239" href="#L239" id="L239">
<i class="fa fa-link"></i>
239
</a>
<a class="diff-line-num" data-line-number="240" href="#L240" id="L240">
<i class="fa fa-link"></i>
240
</a>
<a class="diff-line-num" data-line-number="241" href="#L241" id="L241">
<i class="fa fa-link"></i>
241
</a>
<a class="diff-line-num" data-line-number="242" href="#L242" id="L242">
<i class="fa fa-link"></i>
242
</a>
<a class="diff-line-num" data-line-number="243" href="#L243" id="L243">
<i class="fa fa-link"></i>
243
</a>
<a class="diff-line-num" data-line-number="244" href="#L244" id="L244">
<i class="fa fa-link"></i>
244
</a>
<a class="diff-line-num" data-line-number="245" href="#L245" id="L245">
<i class="fa fa-link"></i>
245
</a>
<a class="diff-line-num" data-line-number="246" href="#L246" id="L246">
<i class="fa fa-link"></i>
246
</a>
<a class="diff-line-num" data-line-number="247" href="#L247" id="L247">
<i class="fa fa-link"></i>
247
</a>
<a class="diff-line-num" data-line-number="248" href="#L248" id="L248">
<i class="fa fa-link"></i>
248
</a>
<a class="diff-line-num" data-line-number="249" href="#L249" id="L249">
<i class="fa fa-link"></i>
249
</a>
<a class="diff-line-num" data-line-number="250" href="#L250" id="L250">
<i class="fa fa-link"></i>
250
</a>
<a class="diff-line-num" data-line-number="251" href="#L251" id="L251">
<i class="fa fa-link"></i>
251
</a>
<a class="diff-line-num" data-line-number="252" href="#L252" id="L252">
<i class="fa fa-link"></i>
252
</a>
<a class="diff-line-num" data-line-number="253" href="#L253" id="L253">
<i class="fa fa-link"></i>
253
</a>
<a class="diff-line-num" data-line-number="254" href="#L254" id="L254">
<i class="fa fa-link"></i>
254
</a>
<a class="diff-line-num" data-line-number="255" href="#L255" id="L255">
<i class="fa fa-link"></i>
255
</a>
<a class="diff-line-num" data-line-number="256" href="#L256" id="L256">
<i class="fa fa-link"></i>
256
</a>
<a class="diff-line-num" data-line-number="257" href="#L257" id="L257">
<i class="fa fa-link"></i>
257
</a>
<a class="diff-line-num" data-line-number="258" href="#L258" id="L258">
<i class="fa fa-link"></i>
258
</a>
<a class="diff-line-num" data-line-number="259" href="#L259" id="L259">
<i class="fa fa-link"></i>
259
</a>
<a class="diff-line-num" data-line-number="260" href="#L260" id="L260">
<i class="fa fa-link"></i>
260
</a>
<a class="diff-line-num" data-line-number="261" href="#L261" id="L261">
<i class="fa fa-link"></i>
261
</a>
<a class="diff-line-num" data-line-number="262" href="#L262" id="L262">
<i class="fa fa-link"></i>
262
</a>
<a class="diff-line-num" data-line-number="263" href="#L263" id="L263">
<i class="fa fa-link"></i>
263
</a>
<a class="diff-line-num" data-line-number="264" href="#L264" id="L264">
<i class="fa fa-link"></i>
264
</a>
<a class="diff-line-num" data-line-number="265" href="#L265" id="L265">
<i class="fa fa-link"></i>
265
</a>
<a class="diff-line-num" data-line-number="266" href="#L266" id="L266">
<i class="fa fa-link"></i>
266
</a>
<a class="diff-line-num" data-line-number="267" href="#L267" id="L267">
<i class="fa fa-link"></i>
267
</a>
<a class="diff-line-num" data-line-number="268" href="#L268" id="L268">
<i class="fa fa-link"></i>
268
</a>
<a class="diff-line-num" data-line-number="269" href="#L269" id="L269">
<i class="fa fa-link"></i>
269
</a>
<a class="diff-line-num" data-line-number="270" href="#L270" id="L270">
<i class="fa fa-link"></i>
270
</a>
<a class="diff-line-num" data-line-number="271" href="#L271" id="L271">
<i class="fa fa-link"></i>
271
</a>
<a class="diff-line-num" data-line-number="272" href="#L272" id="L272">
<i class="fa fa-link"></i>
272
</a>
<a class="diff-line-num" data-line-number="273" href="#L273" id="L273">
<i class="fa fa-link"></i>
273
</a>
<a class="diff-line-num" data-line-number="274" href="#L274" id="L274">
<i class="fa fa-link"></i>
274
</a>
<a class="diff-line-num" data-line-number="275" href="#L275" id="L275">
<i class="fa fa-link"></i>
275
</a>
<a class="diff-line-num" data-line-number="276" href="#L276" id="L276">
<i class="fa fa-link"></i>
276
</a>
<a class="diff-line-num" data-line-number="277" href="#L277" id="L277">
<i class="fa fa-link"></i>
277
</a>
<a class="diff-line-num" data-line-number="278" href="#L278" id="L278">
<i class="fa fa-link"></i>
278
</a>
<a class="diff-line-num" data-line-number="279" href="#L279" id="L279">
<i class="fa fa-link"></i>
279
</a>
<a class="diff-line-num" data-line-number="280" href="#L280" id="L280">
<i class="fa fa-link"></i>
280
</a>
<a class="diff-line-num" data-line-number="281" href="#L281" id="L281">
<i class="fa fa-link"></i>
281
</a>
<a class="diff-line-num" data-line-number="282" href="#L282" id="L282">
<i class="fa fa-link"></i>
282
</a>
<a class="diff-line-num" data-line-number="283" href="#L283" id="L283">
<i class="fa fa-link"></i>
283
</a>
<a class="diff-line-num" data-line-number="284" href="#L284" id="L284">
<i class="fa fa-link"></i>
284
</a>
<a class="diff-line-num" data-line-number="285" href="#L285" id="L285">
<i class="fa fa-link"></i>
285
</a>
<a class="diff-line-num" data-line-number="286" href="#L286" id="L286">
<i class="fa fa-link"></i>
286
</a>
<a class="diff-line-num" data-line-number="287" href="#L287" id="L287">
<i class="fa fa-link"></i>
287
</a>
<a class="diff-line-num" data-line-number="288" href="#L288" id="L288">
<i class="fa fa-link"></i>
288
</a>
<a class="diff-line-num" data-line-number="289" href="#L289" id="L289">
<i class="fa fa-link"></i>
289
</a>
<a class="diff-line-num" data-line-number="290" href="#L290" id="L290">
<i class="fa fa-link"></i>
290
</a>
<a class="diff-line-num" data-line-number="291" href="#L291" id="L291">
<i class="fa fa-link"></i>
291
</a>
<a class="diff-line-num" data-line-number="292" href="#L292" id="L292">
<i class="fa fa-link"></i>
292
</a>
<a class="diff-line-num" data-line-number="293" href="#L293" id="L293">
<i class="fa fa-link"></i>
293
</a>
<a class="diff-line-num" data-line-number="294" href="#L294" id="L294">
<i class="fa fa-link"></i>
294
</a>
<a class="diff-line-num" data-line-number="295" href="#L295" id="L295">
<i class="fa fa-link"></i>
295
</a>
<a class="diff-line-num" data-line-number="296" href="#L296" id="L296">
<i class="fa fa-link"></i>
296
</a>
<a class="diff-line-num" data-line-number="297" href="#L297" id="L297">
<i class="fa fa-link"></i>
297
</a>
<a class="diff-line-num" data-line-number="298" href="#L298" id="L298">
<i class="fa fa-link"></i>
298
</a>
<a class="diff-line-num" data-line-number="299" href="#L299" id="L299">
<i class="fa fa-link"></i>
299
</a>
<a class="diff-line-num" data-line-number="300" href="#L300" id="L300">
<i class="fa fa-link"></i>
300
</a>
<a class="diff-line-num" data-line-number="301" href="#L301" id="L301">
<i class="fa fa-link"></i>
301
</a>
<a class="diff-line-num" data-line-number="302" href="#L302" id="L302">
<i class="fa fa-link"></i>
302
</a>
<a class="diff-line-num" data-line-number="303" href="#L303" id="L303">
<i class="fa fa-link"></i>
303
</a>
<a class="diff-line-num" data-line-number="304" href="#L304" id="L304">
<i class="fa fa-link"></i>
304
</a>
<a class="diff-line-num" data-line-number="305" href="#L305" id="L305">
<i class="fa fa-link"></i>
305
</a>
<a class="diff-line-num" data-line-number="306" href="#L306" id="L306">
<i class="fa fa-link"></i>
306
</a>
<a class="diff-line-num" data-line-number="307" href="#L307" id="L307">
<i class="fa fa-link"></i>
307
</a>
<a class="diff-line-num" data-line-number="308" href="#L308" id="L308">
<i class="fa fa-link"></i>
308
</a>
<a class="diff-line-num" data-line-number="309" href="#L309" id="L309">
<i class="fa fa-link"></i>
309
</a>
<a class="diff-line-num" data-line-number="310" href="#L310" id="L310">
<i class="fa fa-link"></i>
310
</a>
<a class="diff-line-num" data-line-number="311" href="#L311" id="L311">
<i class="fa fa-link"></i>
311
</a>
<a class="diff-line-num" data-line-number="312" href="#L312" id="L312">
<i class="fa fa-link"></i>
312
</a>
<a class="diff-line-num" data-line-number="313" href="#L313" id="L313">
<i class="fa fa-link"></i>
313
</a>
<a class="diff-line-num" data-line-number="314" href="#L314" id="L314">
<i class="fa fa-link"></i>
314
</a>
<a class="diff-line-num" data-line-number="315" href="#L315" id="L315">
<i class="fa fa-link"></i>
315
</a>
<a class="diff-line-num" data-line-number="316" href="#L316" id="L316">
<i class="fa fa-link"></i>
316
</a>
<a class="diff-line-num" data-line-number="317" href="#L317" id="L317">
<i class="fa fa-link"></i>
317
</a>
<a class="diff-line-num" data-line-number="318" href="#L318" id="L318">
<i class="fa fa-link"></i>
318
</a>
<a class="diff-line-num" data-line-number="319" href="#L319" id="L319">
<i class="fa fa-link"></i>
319
</a>
<a class="diff-line-num" data-line-number="320" href="#L320" id="L320">
<i class="fa fa-link"></i>
320
</a>
<a class="diff-line-num" data-line-number="321" href="#L321" id="L321">
<i class="fa fa-link"></i>
321
</a>
<a class="diff-line-num" data-line-number="322" href="#L322" id="L322">
<i class="fa fa-link"></i>
322
</a>
<a class="diff-line-num" data-line-number="323" href="#L323" id="L323">
<i class="fa fa-link"></i>
323
</a>
<a class="diff-line-num" data-line-number="324" href="#L324" id="L324">
<i class="fa fa-link"></i>
324
</a>
<a class="diff-line-num" data-line-number="325" href="#L325" id="L325">
<i class="fa fa-link"></i>
325
</a>
<a class="diff-line-num" data-line-number="326" href="#L326" id="L326">
<i class="fa fa-link"></i>
326
</a>
<a class="diff-line-num" data-line-number="327" href="#L327" id="L327">
<i class="fa fa-link"></i>
327
</a>
<a class="diff-line-num" data-line-number="328" href="#L328" id="L328">
<i class="fa fa-link"></i>
328
</a>
<a class="diff-line-num" data-line-number="329" href="#L329" id="L329">
<i class="fa fa-link"></i>
329
</a>
<a class="diff-line-num" data-line-number="330" href="#L330" id="L330">
<i class="fa fa-link"></i>
330
</a>
<a class="diff-line-num" data-line-number="331" href="#L331" id="L331">
<i class="fa fa-link"></i>
331
</a>
<a class="diff-line-num" data-line-number="332" href="#L332" id="L332">
<i class="fa fa-link"></i>
332
</a>
<a class="diff-line-num" data-line-number="333" href="#L333" id="L333">
<i class="fa fa-link"></i>
333
</a>
<a class="diff-line-num" data-line-number="334" href="#L334" id="L334">
<i class="fa fa-link"></i>
334
</a>
<a class="diff-line-num" data-line-number="335" href="#L335" id="L335">
<i class="fa fa-link"></i>
335
</a>
<a class="diff-line-num" data-line-number="336" href="#L336" id="L336">
<i class="fa fa-link"></i>
336
</a>
<a class="diff-line-num" data-line-number="337" href="#L337" id="L337">
<i class="fa fa-link"></i>
337
</a>
<a class="diff-line-num" data-line-number="338" href="#L338" id="L338">
<i class="fa fa-link"></i>
338
</a>
<a class="diff-line-num" data-line-number="339" href="#L339" id="L339">
<i class="fa fa-link"></i>
339
</a>
<a class="diff-line-num" data-line-number="340" href="#L340" id="L340">
<i class="fa fa-link"></i>
340
</a>
<a class="diff-line-num" data-line-number="341" href="#L341" id="L341">
<i class="fa fa-link"></i>
341
</a>
<a class="diff-line-num" data-line-number="342" href="#L342" id="L342">
<i class="fa fa-link"></i>
342
</a>
<a class="diff-line-num" data-line-number="343" href="#L343" id="L343">
<i class="fa fa-link"></i>
343
</a>
<a class="diff-line-num" data-line-number="344" href="#L344" id="L344">
<i class="fa fa-link"></i>
344
</a>
<a class="diff-line-num" data-line-number="345" href="#L345" id="L345">
<i class="fa fa-link"></i>
345
</a>
<a class="diff-line-num" data-line-number="346" href="#L346" id="L346">
<i class="fa fa-link"></i>
346
</a>
<a class="diff-line-num" data-line-number="347" href="#L347" id="L347">
<i class="fa fa-link"></i>
347
</a>
<a class="diff-line-num" data-line-number="348" href="#L348" id="L348">
<i class="fa fa-link"></i>
348
</a>
<a class="diff-line-num" data-line-number="349" href="#L349" id="L349">
<i class="fa fa-link"></i>
349
</a>
<a class="diff-line-num" data-line-number="350" href="#L350" id="L350">
<i class="fa fa-link"></i>
350
</a>
<a class="diff-line-num" data-line-number="351" href="#L351" id="L351">
<i class="fa fa-link"></i>
351
</a>
<a class="diff-line-num" data-line-number="352" href="#L352" id="L352">
<i class="fa fa-link"></i>
352
</a>
<a class="diff-line-num" data-line-number="353" href="#L353" id="L353">
<i class="fa fa-link"></i>
353
</a>
<a class="diff-line-num" data-line-number="354" href="#L354" id="L354">
<i class="fa fa-link"></i>
354
</a>
<a class="diff-line-num" data-line-number="355" href="#L355" id="L355">
<i class="fa fa-link"></i>
355
</a>
<a class="diff-line-num" data-line-number="356" href="#L356" id="L356">
<i class="fa fa-link"></i>
356
</a>
<a class="diff-line-num" data-line-number="357" href="#L357" id="L357">
<i class="fa fa-link"></i>
357
</a>
<a class="diff-line-num" data-line-number="358" href="#L358" id="L358">
<i class="fa fa-link"></i>
358
</a>
<a class="diff-line-num" data-line-number="359" href="#L359" id="L359">
<i class="fa fa-link"></i>
359
</a>
<a class="diff-line-num" data-line-number="360" href="#L360" id="L360">
<i class="fa fa-link"></i>
360
</a>
<a class="diff-line-num" data-line-number="361" href="#L361" id="L361">
<i class="fa fa-link"></i>
361
</a>
<a class="diff-line-num" data-line-number="362" href="#L362" id="L362">
<i class="fa fa-link"></i>
362
</a>
<a class="diff-line-num" data-line-number="363" href="#L363" id="L363">
<i class="fa fa-link"></i>
363
</a>
<a class="diff-line-num" data-line-number="364" href="#L364" id="L364">
<i class="fa fa-link"></i>
364
</a>
<a class="diff-line-num" data-line-number="365" href="#L365" id="L365">
<i class="fa fa-link"></i>
365
</a>
<a class="diff-line-num" data-line-number="366" href="#L366" id="L366">
<i class="fa fa-link"></i>
366
</a>
<a class="diff-line-num" data-line-number="367" href="#L367" id="L367">
<i class="fa fa-link"></i>
367
</a>
<a class="diff-line-num" data-line-number="368" href="#L368" id="L368">
<i class="fa fa-link"></i>
368
</a>
<a class="diff-line-num" data-line-number="369" href="#L369" id="L369">
<i class="fa fa-link"></i>
369
</a>
<a class="diff-line-num" data-line-number="370" href="#L370" id="L370">
<i class="fa fa-link"></i>
370
</a>
<a class="diff-line-num" data-line-number="371" href="#L371" id="L371">
<i class="fa fa-link"></i>
371
</a>
<a class="diff-line-num" data-line-number="372" href="#L372" id="L372">
<i class="fa fa-link"></i>
372
</a>
<a class="diff-line-num" data-line-number="373" href="#L373" id="L373">
<i class="fa fa-link"></i>
373
</a>
<a class="diff-line-num" data-line-number="374" href="#L374" id="L374">
<i class="fa fa-link"></i>
374
</a>
<a class="diff-line-num" data-line-number="375" href="#L375" id="L375">
<i class="fa fa-link"></i>
375
</a>
<a class="diff-line-num" data-line-number="376" href="#L376" id="L376">
<i class="fa fa-link"></i>
376
</a>
<a class="diff-line-num" data-line-number="377" href="#L377" id="L377">
<i class="fa fa-link"></i>
377
</a>
<a class="diff-line-num" data-line-number="378" href="#L378" id="L378">
<i class="fa fa-link"></i>
378
</a>
<a class="diff-line-num" data-line-number="379" href="#L379" id="L379">
<i class="fa fa-link"></i>
379
</a>
<a class="diff-line-num" data-line-number="380" href="#L380" id="L380">
<i class="fa fa-link"></i>
380
</a>
<a class="diff-line-num" data-line-number="381" href="#L381" id="L381">
<i class="fa fa-link"></i>
381
</a>
<a class="diff-line-num" data-line-number="382" href="#L382" id="L382">
<i class="fa fa-link"></i>
382
</a>
<a class="diff-line-num" data-line-number="383" href="#L383" id="L383">
<i class="fa fa-link"></i>
383
</a>
<a class="diff-line-num" data-line-number="384" href="#L384" id="L384">
<i class="fa fa-link"></i>
384
</a>
<a class="diff-line-num" data-line-number="385" href="#L385" id="L385">
<i class="fa fa-link"></i>
385
</a>
<a class="diff-line-num" data-line-number="386" href="#L386" id="L386">
<i class="fa fa-link"></i>
386
</a>
<a class="diff-line-num" data-line-number="387" href="#L387" id="L387">
<i class="fa fa-link"></i>
387
</a>
<a class="diff-line-num" data-line-number="388" href="#L388" id="L388">
<i class="fa fa-link"></i>
388
</a>
<a class="diff-line-num" data-line-number="389" href="#L389" id="L389">
<i class="fa fa-link"></i>
389
</a>
<a class="diff-line-num" data-line-number="390" href="#L390" id="L390">
<i class="fa fa-link"></i>
390
</a>
<a class="diff-line-num" data-line-number="391" href="#L391" id="L391">
<i class="fa fa-link"></i>
391
</a>
<a class="diff-line-num" data-line-number="392" href="#L392" id="L392">
<i class="fa fa-link"></i>
392
</a>
<a class="diff-line-num" data-line-number="393" href="#L393" id="L393">
<i class="fa fa-link"></i>
393
</a>
<a class="diff-line-num" data-line-number="394" href="#L394" id="L394">
<i class="fa fa-link"></i>
394
</a>
<a class="diff-line-num" data-line-number="395" href="#L395" id="L395">
<i class="fa fa-link"></i>
395
</a>
<a class="diff-line-num" data-line-number="396" href="#L396" id="L396">
<i class="fa fa-link"></i>
396
</a>
<a class="diff-line-num" data-line-number="397" href="#L397" id="L397">
<i class="fa fa-link"></i>
397
</a>
<a class="diff-line-num" data-line-number="398" href="#L398" id="L398">
<i class="fa fa-link"></i>
398
</a>
<a class="diff-line-num" data-line-number="399" href="#L399" id="L399">
<i class="fa fa-link"></i>
399
</a>
<a class="diff-line-num" data-line-number="400" href="#L400" id="L400">
<i class="fa fa-link"></i>
400
</a>
<a class="diff-line-num" data-line-number="401" href="#L401" id="L401">
<i class="fa fa-link"></i>
401
</a>
<a class="diff-line-num" data-line-number="402" href="#L402" id="L402">
<i class="fa fa-link"></i>
402
</a>
<a class="diff-line-num" data-line-number="403" href="#L403" id="L403">
<i class="fa fa-link"></i>
403
</a>
<a class="diff-line-num" data-line-number="404" href="#L404" id="L404">
<i class="fa fa-link"></i>
404
</a>
<a class="diff-line-num" data-line-number="405" href="#L405" id="L405">
<i class="fa fa-link"></i>
405
</a>
<a class="diff-line-num" data-line-number="406" href="#L406" id="L406">
<i class="fa fa-link"></i>
406
</a>
<a class="diff-line-num" data-line-number="407" href="#L407" id="L407">
<i class="fa fa-link"></i>
407
</a>
<a class="diff-line-num" data-line-number="408" href="#L408" id="L408">
<i class="fa fa-link"></i>
408
</a>
<a class="diff-line-num" data-line-number="409" href="#L409" id="L409">
<i class="fa fa-link"></i>
409
</a>
<a class="diff-line-num" data-line-number="410" href="#L410" id="L410">
<i class="fa fa-link"></i>
410
</a>
<a class="diff-line-num" data-line-number="411" href="#L411" id="L411">
<i class="fa fa-link"></i>
411
</a>
<a class="diff-line-num" data-line-number="412" href="#L412" id="L412">
<i class="fa fa-link"></i>
412
</a>
<a class="diff-line-num" data-line-number="413" href="#L413" id="L413">
<i class="fa fa-link"></i>
413
</a>
<a class="diff-line-num" data-line-number="414" href="#L414" id="L414">
<i class="fa fa-link"></i>
414
</a>
<a class="diff-line-num" data-line-number="415" href="#L415" id="L415">
<i class="fa fa-link"></i>
415
</a>
<a class="diff-line-num" data-line-number="416" href="#L416" id="L416">
<i class="fa fa-link"></i>
416
</a>
<a class="diff-line-num" data-line-number="417" href="#L417" id="L417">
<i class="fa fa-link"></i>
417
</a>
<a class="diff-line-num" data-line-number="418" href="#L418" id="L418">
<i class="fa fa-link"></i>
418
</a>
<a class="diff-line-num" data-line-number="419" href="#L419" id="L419">
<i class="fa fa-link"></i>
419
</a>
<a class="diff-line-num" data-line-number="420" href="#L420" id="L420">
<i class="fa fa-link"></i>
420
</a>
<a class="diff-line-num" data-line-number="421" href="#L421" id="L421">
<i class="fa fa-link"></i>
421
</a>
<a class="diff-line-num" data-line-number="422" href="#L422" id="L422">
<i class="fa fa-link"></i>
422
</a>
<a class="diff-line-num" data-line-number="423" href="#L423" id="L423">
<i class="fa fa-link"></i>
423
</a>
<a class="diff-line-num" data-line-number="424" href="#L424" id="L424">
<i class="fa fa-link"></i>
424
</a>
<a class="diff-line-num" data-line-number="425" href="#L425" id="L425">
<i class="fa fa-link"></i>
425
</a>
<a class="diff-line-num" data-line-number="426" href="#L426" id="L426">
<i class="fa fa-link"></i>
426
</a>
<a class="diff-line-num" data-line-number="427" href="#L427" id="L427">
<i class="fa fa-link"></i>
427
</a>
<a class="diff-line-num" data-line-number="428" href="#L428" id="L428">
<i class="fa fa-link"></i>
428
</a>
<a class="diff-line-num" data-line-number="429" href="#L429" id="L429">
<i class="fa fa-link"></i>
429
</a>
<a class="diff-line-num" data-line-number="430" href="#L430" id="L430">
<i class="fa fa-link"></i>
430
</a>
<a class="diff-line-num" data-line-number="431" href="#L431" id="L431">
<i class="fa fa-link"></i>
431
</a>
<a class="diff-line-num" data-line-number="432" href="#L432" id="L432">
<i class="fa fa-link"></i>
432
</a>
<a class="diff-line-num" data-line-number="433" href="#L433" id="L433">
<i class="fa fa-link"></i>
433
</a>
<a class="diff-line-num" data-line-number="434" href="#L434" id="L434">
<i class="fa fa-link"></i>
434
</a>
<a class="diff-line-num" data-line-number="435" href="#L435" id="L435">
<i class="fa fa-link"></i>
435
</a>
<a class="diff-line-num" data-line-number="436" href="#L436" id="L436">
<i class="fa fa-link"></i>
436
</a>
<a class="diff-line-num" data-line-number="437" href="#L437" id="L437">
<i class="fa fa-link"></i>
437
</a>
<a class="diff-line-num" data-line-number="438" href="#L438" id="L438">
<i class="fa fa-link"></i>
438
</a>
<a class="diff-line-num" data-line-number="439" href="#L439" id="L439">
<i class="fa fa-link"></i>
439
</a>
<a class="diff-line-num" data-line-number="440" href="#L440" id="L440">
<i class="fa fa-link"></i>
440
</a>
<a class="diff-line-num" data-line-number="441" href="#L441" id="L441">
<i class="fa fa-link"></i>
441
</a>
<a class="diff-line-num" data-line-number="442" href="#L442" id="L442">
<i class="fa fa-link"></i>
442
</a>
<a class="diff-line-num" data-line-number="443" href="#L443" id="L443">
<i class="fa fa-link"></i>
443
</a>
<a class="diff-line-num" data-line-number="444" href="#L444" id="L444">
<i class="fa fa-link"></i>
444
</a>
<a class="diff-line-num" data-line-number="445" href="#L445" id="L445">
<i class="fa fa-link"></i>
445
</a>
<a class="diff-line-num" data-line-number="446" href="#L446" id="L446">
<i class="fa fa-link"></i>
446
</a>
<a class="diff-line-num" data-line-number="447" href="#L447" id="L447">
<i class="fa fa-link"></i>
447
</a>
<a class="diff-line-num" data-line-number="448" href="#L448" id="L448">
<i class="fa fa-link"></i>
448
</a>
<a class="diff-line-num" data-line-number="449" href="#L449" id="L449">
<i class="fa fa-link"></i>
449
</a>
<a class="diff-line-num" data-line-number="450" href="#L450" id="L450">
<i class="fa fa-link"></i>
450
</a>
<a class="diff-line-num" data-line-number="451" href="#L451" id="L451">
<i class="fa fa-link"></i>
451
</a>
<a class="diff-line-num" data-line-number="452" href="#L452" id="L452">
<i class="fa fa-link"></i>
452
</a>
<a class="diff-line-num" data-line-number="453" href="#L453" id="L453">
<i class="fa fa-link"></i>
453
</a>
<a class="diff-line-num" data-line-number="454" href="#L454" id="L454">
<i class="fa fa-link"></i>
454
</a>
<a class="diff-line-num" data-line-number="455" href="#L455" id="L455">
<i class="fa fa-link"></i>
455
</a>
<a class="diff-line-num" data-line-number="456" href="#L456" id="L456">
<i class="fa fa-link"></i>
456
</a>
<a class="diff-line-num" data-line-number="457" href="#L457" id="L457">
<i class="fa fa-link"></i>
457
</a>
<a class="diff-line-num" data-line-number="458" href="#L458" id="L458">
<i class="fa fa-link"></i>
458
</a>
<a class="diff-line-num" data-line-number="459" href="#L459" id="L459">
<i class="fa fa-link"></i>
459
</a>
<a class="diff-line-num" data-line-number="460" href="#L460" id="L460">
<i class="fa fa-link"></i>
460
</a>
<a class="diff-line-num" data-line-number="461" href="#L461" id="L461">
<i class="fa fa-link"></i>
461
</a>
<a class="diff-line-num" data-line-number="462" href="#L462" id="L462">
<i class="fa fa-link"></i>
462
</a>
<a class="diff-line-num" data-line-number="463" href="#L463" id="L463">
<i class="fa fa-link"></i>
463
</a>
<a class="diff-line-num" data-line-number="464" href="#L464" id="L464">
<i class="fa fa-link"></i>
464
</a>
<a class="diff-line-num" data-line-number="465" href="#L465" id="L465">
<i class="fa fa-link"></i>
465
</a>
<a class="diff-line-num" data-line-number="466" href="#L466" id="L466">
<i class="fa fa-link"></i>
466
</a>
<a class="diff-line-num" data-line-number="467" href="#L467" id="L467">
<i class="fa fa-link"></i>
467
</a>
<a class="diff-line-num" data-line-number="468" href="#L468" id="L468">
<i class="fa fa-link"></i>
468
</a>
<a class="diff-line-num" data-line-number="469" href="#L469" id="L469">
<i class="fa fa-link"></i>
469
</a>
<a class="diff-line-num" data-line-number="470" href="#L470" id="L470">
<i class="fa fa-link"></i>
470
</a>
<a class="diff-line-num" data-line-number="471" href="#L471" id="L471">
<i class="fa fa-link"></i>
471
</a>
<a class="diff-line-num" data-line-number="472" href="#L472" id="L472">
<i class="fa fa-link"></i>
472
</a>
<a class="diff-line-num" data-line-number="473" href="#L473" id="L473">
<i class="fa fa-link"></i>
473
</a>
<a class="diff-line-num" data-line-number="474" href="#L474" id="L474">
<i class="fa fa-link"></i>
474
</a>
<a class="diff-line-num" data-line-number="475" href="#L475" id="L475">
<i class="fa fa-link"></i>
475
</a>
<a class="diff-line-num" data-line-number="476" href="#L476" id="L476">
<i class="fa fa-link"></i>
476
</a>
<a class="diff-line-num" data-line-number="477" href="#L477" id="L477">
<i class="fa fa-link"></i>
477
</a>
<a class="diff-line-num" data-line-number="478" href="#L478" id="L478">
<i class="fa fa-link"></i>
478
</a>
<a class="diff-line-num" data-line-number="479" href="#L479" id="L479">
<i class="fa fa-link"></i>
479
</a>
<a class="diff-line-num" data-line-number="480" href="#L480" id="L480">
<i class="fa fa-link"></i>
480
</a>
<a class="diff-line-num" data-line-number="481" href="#L481" id="L481">
<i class="fa fa-link"></i>
481
</a>
<a class="diff-line-num" data-line-number="482" href="#L482" id="L482">
<i class="fa fa-link"></i>
482
</a>
<a class="diff-line-num" data-line-number="483" href="#L483" id="L483">
<i class="fa fa-link"></i>
483
</a>
<a class="diff-line-num" data-line-number="484" href="#L484" id="L484">
<i class="fa fa-link"></i>
484
</a>
<a class="diff-line-num" data-line-number="485" href="#L485" id="L485">
<i class="fa fa-link"></i>
485
</a>
<a class="diff-line-num" data-line-number="486" href="#L486" id="L486">
<i class="fa fa-link"></i>
486
</a>
<a class="diff-line-num" data-line-number="487" href="#L487" id="L487">
<i class="fa fa-link"></i>
487
</a>
<a class="diff-line-num" data-line-number="488" href="#L488" id="L488">
<i class="fa fa-link"></i>
488
</a>
<a class="diff-line-num" data-line-number="489" href="#L489" id="L489">
<i class="fa fa-link"></i>
489
</a>
<a class="diff-line-num" data-line-number="490" href="#L490" id="L490">
<i class="fa fa-link"></i>
490
</a>
<a class="diff-line-num" data-line-number="491" href="#L491" id="L491">
<i class="fa fa-link"></i>
491
</a>
<a class="diff-line-num" data-line-number="492" href="#L492" id="L492">
<i class="fa fa-link"></i>
492
</a>
<a class="diff-line-num" data-line-number="493" href="#L493" id="L493">
<i class="fa fa-link"></i>
493
</a>
<a class="diff-line-num" data-line-number="494" href="#L494" id="L494">
<i class="fa fa-link"></i>
494
</a>
<a class="diff-line-num" data-line-number="495" href="#L495" id="L495">
<i class="fa fa-link"></i>
495
</a>
<a class="diff-line-num" data-line-number="496" href="#L496" id="L496">
<i class="fa fa-link"></i>
496
</a>
<a class="diff-line-num" data-line-number="497" href="#L497" id="L497">
<i class="fa fa-link"></i>
497
</a>
<a class="diff-line-num" data-line-number="498" href="#L498" id="L498">
<i class="fa fa-link"></i>
498
</a>
<a class="diff-line-num" data-line-number="499" href="#L499" id="L499">
<i class="fa fa-link"></i>
499
</a>
<a class="diff-line-num" data-line-number="500" href="#L500" id="L500">
<i class="fa fa-link"></i>
500
</a>
<a class="diff-line-num" data-line-number="501" href="#L501" id="L501">
<i class="fa fa-link"></i>
501
</a>
<a class="diff-line-num" data-line-number="502" href="#L502" id="L502">
<i class="fa fa-link"></i>
502
</a>
<a class="diff-line-num" data-line-number="503" href="#L503" id="L503">
<i class="fa fa-link"></i>
503
</a>
<a class="diff-line-num" data-line-number="504" href="#L504" id="L504">
<i class="fa fa-link"></i>
504
</a>
<a class="diff-line-num" data-line-number="505" href="#L505" id="L505">
<i class="fa fa-link"></i>
505
</a>
<a class="diff-line-num" data-line-number="506" href="#L506" id="L506">
<i class="fa fa-link"></i>
506
</a>
<a class="diff-line-num" data-line-number="507" href="#L507" id="L507">
<i class="fa fa-link"></i>
507
</a>
<a class="diff-line-num" data-line-number="508" href="#L508" id="L508">
<i class="fa fa-link"></i>
508
</a>
<a class="diff-line-num" data-line-number="509" href="#L509" id="L509">
<i class="fa fa-link"></i>
509
</a>
<a class="diff-line-num" data-line-number="510" href="#L510" id="L510">
<i class="fa fa-link"></i>
510
</a>
<a class="diff-line-num" data-line-number="511" href="#L511" id="L511">
<i class="fa fa-link"></i>
511
</a>
<a class="diff-line-num" data-line-number="512" href="#L512" id="L512">
<i class="fa fa-link"></i>
512
</a>
<a class="diff-line-num" data-line-number="513" href="#L513" id="L513">
<i class="fa fa-link"></i>
513
</a>
<a class="diff-line-num" data-line-number="514" href="#L514" id="L514">
<i class="fa fa-link"></i>
514
</a>
<a class="diff-line-num" data-line-number="515" href="#L515" id="L515">
<i class="fa fa-link"></i>
515
</a>
<a class="diff-line-num" data-line-number="516" href="#L516" id="L516">
<i class="fa fa-link"></i>
516
</a>
<a class="diff-line-num" data-line-number="517" href="#L517" id="L517">
<i class="fa fa-link"></i>
517
</a>
<a class="diff-line-num" data-line-number="518" href="#L518" id="L518">
<i class="fa fa-link"></i>
518
</a>
<a class="diff-line-num" data-line-number="519" href="#L519" id="L519">
<i class="fa fa-link"></i>
519
</a>
<a class="diff-line-num" data-line-number="520" href="#L520" id="L520">
<i class="fa fa-link"></i>
520
</a>
<a class="diff-line-num" data-line-number="521" href="#L521" id="L521">
<i class="fa fa-link"></i>
521
</a>
<a class="diff-line-num" data-line-number="522" href="#L522" id="L522">
<i class="fa fa-link"></i>
522
</a>
<a class="diff-line-num" data-line-number="523" href="#L523" id="L523">
<i class="fa fa-link"></i>
523
</a>
<a class="diff-line-num" data-line-number="524" href="#L524" id="L524">
<i class="fa fa-link"></i>
524
</a>
<a class="diff-line-num" data-line-number="525" href="#L525" id="L525">
<i class="fa fa-link"></i>
525
</a>
<a class="diff-line-num" data-line-number="526" href="#L526" id="L526">
<i class="fa fa-link"></i>
526
</a>
<a class="diff-line-num" data-line-number="527" href="#L527" id="L527">
<i class="fa fa-link"></i>
527
</a>
<a class="diff-line-num" data-line-number="528" href="#L528" id="L528">
<i class="fa fa-link"></i>
528
</a>
<a class="diff-line-num" data-line-number="529" href="#L529" id="L529">
<i class="fa fa-link"></i>
529
</a>
<a class="diff-line-num" data-line-number="530" href="#L530" id="L530">
<i class="fa fa-link"></i>
530
</a>
<a class="diff-line-num" data-line-number="531" href="#L531" id="L531">
<i class="fa fa-link"></i>
531
</a>
<a class="diff-line-num" data-line-number="532" href="#L532" id="L532">
<i class="fa fa-link"></i>
532
</a>
<a class="diff-line-num" data-line-number="533" href="#L533" id="L533">
<i class="fa fa-link"></i>
533
</a>
<a class="diff-line-num" data-line-number="534" href="#L534" id="L534">
<i class="fa fa-link"></i>
534
</a>
<a class="diff-line-num" data-line-number="535" href="#L535" id="L535">
<i class="fa fa-link"></i>
535
</a>
<a class="diff-line-num" data-line-number="536" href="#L536" id="L536">
<i class="fa fa-link"></i>
536
</a>
<a class="diff-line-num" data-line-number="537" href="#L537" id="L537">
<i class="fa fa-link"></i>
537
</a>
<a class="diff-line-num" data-line-number="538" href="#L538" id="L538">
<i class="fa fa-link"></i>
538
</a>
<a class="diff-line-num" data-line-number="539" href="#L539" id="L539">
<i class="fa fa-link"></i>
539
</a>
<a class="diff-line-num" data-line-number="540" href="#L540" id="L540">
<i class="fa fa-link"></i>
540
</a>
<a class="diff-line-num" data-line-number="541" href="#L541" id="L541">
<i class="fa fa-link"></i>
541
</a>
<a class="diff-line-num" data-line-number="542" href="#L542" id="L542">
<i class="fa fa-link"></i>
542
</a>
<a class="diff-line-num" data-line-number="543" href="#L543" id="L543">
<i class="fa fa-link"></i>
543
</a>
<a class="diff-line-num" data-line-number="544" href="#L544" id="L544">
<i class="fa fa-link"></i>
544
</a>
<a class="diff-line-num" data-line-number="545" href="#L545" id="L545">
<i class="fa fa-link"></i>
545
</a>
<a class="diff-line-num" data-line-number="546" href="#L546" id="L546">
<i class="fa fa-link"></i>
546
</a>
<a class="diff-line-num" data-line-number="547" href="#L547" id="L547">
<i class="fa fa-link"></i>
547
</a>
<a class="diff-line-num" data-line-number="548" href="#L548" id="L548">
<i class="fa fa-link"></i>
548
</a>
<a class="diff-line-num" data-line-number="549" href="#L549" id="L549">
<i class="fa fa-link"></i>
549
</a>
<a class="diff-line-num" data-line-number="550" href="#L550" id="L550">
<i class="fa fa-link"></i>
550
</a>
<a class="diff-line-num" data-line-number="551" href="#L551" id="L551">
<i class="fa fa-link"></i>
551
</a>
<a class="diff-line-num" data-line-number="552" href="#L552" id="L552">
<i class="fa fa-link"></i>
552
</a>
<a class="diff-line-num" data-line-number="553" href="#L553" id="L553">
<i class="fa fa-link"></i>
553
</a>
<a class="diff-line-num" data-line-number="554" href="#L554" id="L554">
<i class="fa fa-link"></i>
554
</a>
<a class="diff-line-num" data-line-number="555" href="#L555" id="L555">
<i class="fa fa-link"></i>
555
</a>
<a class="diff-line-num" data-line-number="556" href="#L556" id="L556">
<i class="fa fa-link"></i>
556
</a>
<a class="diff-line-num" data-line-number="557" href="#L557" id="L557">
<i class="fa fa-link"></i>
557
</a>
<a class="diff-line-num" data-line-number="558" href="#L558" id="L558">
<i class="fa fa-link"></i>
558
</a>
<a class="diff-line-num" data-line-number="559" href="#L559" id="L559">
<i class="fa fa-link"></i>
559
</a>
<a class="diff-line-num" data-line-number="560" href="#L560" id="L560">
<i class="fa fa-link"></i>
560
</a>
<a class="diff-line-num" data-line-number="561" href="#L561" id="L561">
<i class="fa fa-link"></i>
561
</a>
<a class="diff-line-num" data-line-number="562" href="#L562" id="L562">
<i class="fa fa-link"></i>
562
</a>
<a class="diff-line-num" data-line-number="563" href="#L563" id="L563">
<i class="fa fa-link"></i>
563
</a>
<a class="diff-line-num" data-line-number="564" href="#L564" id="L564">
<i class="fa fa-link"></i>
564
</a>
<a class="diff-line-num" data-line-number="565" href="#L565" id="L565">
<i class="fa fa-link"></i>
565
</a>
<a class="diff-line-num" data-line-number="566" href="#L566" id="L566">
<i class="fa fa-link"></i>
566
</a>
<a class="diff-line-num" data-line-number="567" href="#L567" id="L567">
<i class="fa fa-link"></i>
567
</a>
<a class="diff-line-num" data-line-number="568" href="#L568" id="L568">
<i class="fa fa-link"></i>
568
</a>
<a class="diff-line-num" data-line-number="569" href="#L569" id="L569">
<i class="fa fa-link"></i>
569
</a>
<a class="diff-line-num" data-line-number="570" href="#L570" id="L570">
<i class="fa fa-link"></i>
570
</a>
<a class="diff-line-num" data-line-number="571" href="#L571" id="L571">
<i class="fa fa-link"></i>
571
</a>
<a class="diff-line-num" data-line-number="572" href="#L572" id="L572">
<i class="fa fa-link"></i>
572
</a>
<a class="diff-line-num" data-line-number="573" href="#L573" id="L573">
<i class="fa fa-link"></i>
573
</a>
<a class="diff-line-num" data-line-number="574" href="#L574" id="L574">
<i class="fa fa-link"></i>
574
</a>
<a class="diff-line-num" data-line-number="575" href="#L575" id="L575">
<i class="fa fa-link"></i>
575
</a>
<a class="diff-line-num" data-line-number="576" href="#L576" id="L576">
<i class="fa fa-link"></i>
576
</a>
<a class="diff-line-num" data-line-number="577" href="#L577" id="L577">
<i class="fa fa-link"></i>
577
</a>
<a class="diff-line-num" data-line-number="578" href="#L578" id="L578">
<i class="fa fa-link"></i>
578
</a>
<a class="diff-line-num" data-line-number="579" href="#L579" id="L579">
<i class="fa fa-link"></i>
579
</a>
<a class="diff-line-num" data-line-number="580" href="#L580" id="L580">
<i class="fa fa-link"></i>
580
</a>
<a class="diff-line-num" data-line-number="581" href="#L581" id="L581">
<i class="fa fa-link"></i>
581
</a>
<a class="diff-line-num" data-line-number="582" href="#L582" id="L582">
<i class="fa fa-link"></i>
582
</a>
<a class="diff-line-num" data-line-number="583" href="#L583" id="L583">
<i class="fa fa-link"></i>
583
</a>
<a class="diff-line-num" data-line-number="584" href="#L584" id="L584">
<i class="fa fa-link"></i>
584
</a>
<a class="diff-line-num" data-line-number="585" href="#L585" id="L585">
<i class="fa fa-link"></i>
585
</a>
<a class="diff-line-num" data-line-number="586" href="#L586" id="L586">
<i class="fa fa-link"></i>
586
</a>
<a class="diff-line-num" data-line-number="587" href="#L587" id="L587">
<i class="fa fa-link"></i>
587
</a>
<a class="diff-line-num" data-line-number="588" href="#L588" id="L588">
<i class="fa fa-link"></i>
588
</a>
<a class="diff-line-num" data-line-number="589" href="#L589" id="L589">
<i class="fa fa-link"></i>
589
</a>
<a class="diff-line-num" data-line-number="590" href="#L590" id="L590">
<i class="fa fa-link"></i>
590
</a>
<a class="diff-line-num" data-line-number="591" href="#L591" id="L591">
<i class="fa fa-link"></i>
591
</a>
<a class="diff-line-num" data-line-number="592" href="#L592" id="L592">
<i class="fa fa-link"></i>
592
</a>
<a class="diff-line-num" data-line-number="593" href="#L593" id="L593">
<i class="fa fa-link"></i>
593
</a>
<a class="diff-line-num" data-line-number="594" href="#L594" id="L594">
<i class="fa fa-link"></i>
594
</a>
<a class="diff-line-num" data-line-number="595" href="#L595" id="L595">
<i class="fa fa-link"></i>
595
</a>
<a class="diff-line-num" data-line-number="596" href="#L596" id="L596">
<i class="fa fa-link"></i>
596
</a>
<a class="diff-line-num" data-line-number="597" href="#L597" id="L597">
<i class="fa fa-link"></i>
597
</a>
<a class="diff-line-num" data-line-number="598" href="#L598" id="L598">
<i class="fa fa-link"></i>
598
</a>
<a class="diff-line-num" data-line-number="599" href="#L599" id="L599">
<i class="fa fa-link"></i>
599
</a>
<a class="diff-line-num" data-line-number="600" href="#L600" id="L600">
<i class="fa fa-link"></i>
600
</a>
<a class="diff-line-num" data-line-number="601" href="#L601" id="L601">
<i class="fa fa-link"></i>
601
</a>
<a class="diff-line-num" data-line-number="602" href="#L602" id="L602">
<i class="fa fa-link"></i>
602
</a>
<a class="diff-line-num" data-line-number="603" href="#L603" id="L603">
<i class="fa fa-link"></i>
603
</a>
<a class="diff-line-num" data-line-number="604" href="#L604" id="L604">
<i class="fa fa-link"></i>
604
</a>
<a class="diff-line-num" data-line-number="605" href="#L605" id="L605">
<i class="fa fa-link"></i>
605
</a>
<a class="diff-line-num" data-line-number="606" href="#L606" id="L606">
<i class="fa fa-link"></i>
606
</a>
<a class="diff-line-num" data-line-number="607" href="#L607" id="L607">
<i class="fa fa-link"></i>
607
</a>
<a class="diff-line-num" data-line-number="608" href="#L608" id="L608">
<i class="fa fa-link"></i>
608
</a>
<a class="diff-line-num" data-line-number="609" href="#L609" id="L609">
<i class="fa fa-link"></i>
609
</a>
<a class="diff-line-num" data-line-number="610" href="#L610" id="L610">
<i class="fa fa-link"></i>
610
</a>
<a class="diff-line-num" data-line-number="611" href="#L611" id="L611">
<i class="fa fa-link"></i>
611
</a>
<a class="diff-line-num" data-line-number="612" href="#L612" id="L612">
<i class="fa fa-link"></i>
612
</a>
<a class="diff-line-num" data-line-number="613" href="#L613" id="L613">
<i class="fa fa-link"></i>
613
</a>
<a class="diff-line-num" data-line-number="614" href="#L614" id="L614">
<i class="fa fa-link"></i>
614
</a>
<a class="diff-line-num" data-line-number="615" href="#L615" id="L615">
<i class="fa fa-link"></i>
615
</a>
<a class="diff-line-num" data-line-number="616" href="#L616" id="L616">
<i class="fa fa-link"></i>
616
</a>
<a class="diff-line-num" data-line-number="617" href="#L617" id="L617">
<i class="fa fa-link"></i>
617
</a>
<a class="diff-line-num" data-line-number="618" href="#L618" id="L618">
<i class="fa fa-link"></i>
618
</a>
<a class="diff-line-num" data-line-number="619" href="#L619" id="L619">
<i class="fa fa-link"></i>
619
</a>
<a class="diff-line-num" data-line-number="620" href="#L620" id="L620">
<i class="fa fa-link"></i>
620
</a>
<a class="diff-line-num" data-line-number="621" href="#L621" id="L621">
<i class="fa fa-link"></i>
621
</a>
<a class="diff-line-num" data-line-number="622" href="#L622" id="L622">
<i class="fa fa-link"></i>
622
</a>
<a class="diff-line-num" data-line-number="623" href="#L623" id="L623">
<i class="fa fa-link"></i>
623
</a>
<a class="diff-line-num" data-line-number="624" href="#L624" id="L624">
<i class="fa fa-link"></i>
624
</a>
<a class="diff-line-num" data-line-number="625" href="#L625" id="L625">
<i class="fa fa-link"></i>
625
</a>
<a class="diff-line-num" data-line-number="626" href="#L626" id="L626">
<i class="fa fa-link"></i>
626
</a>
<a class="diff-line-num" data-line-number="627" href="#L627" id="L627">
<i class="fa fa-link"></i>
627
</a>
<a class="diff-line-num" data-line-number="628" href="#L628" id="L628">
<i class="fa fa-link"></i>
628
</a>
<a class="diff-line-num" data-line-number="629" href="#L629" id="L629">
<i class="fa fa-link"></i>
629
</a>
<a class="diff-line-num" data-line-number="630" href="#L630" id="L630">
<i class="fa fa-link"></i>
630
</a>
<a class="diff-line-num" data-line-number="631" href="#L631" id="L631">
<i class="fa fa-link"></i>
631
</a>
<a class="diff-line-num" data-line-number="632" href="#L632" id="L632">
<i class="fa fa-link"></i>
632
</a>
<a class="diff-line-num" data-line-number="633" href="#L633" id="L633">
<i class="fa fa-link"></i>
633
</a>
<a class="diff-line-num" data-line-number="634" href="#L634" id="L634">
<i class="fa fa-link"></i>
634
</a>
<a class="diff-line-num" data-line-number="635" href="#L635" id="L635">
<i class="fa fa-link"></i>
635
</a>
<a class="diff-line-num" data-line-number="636" href="#L636" id="L636">
<i class="fa fa-link"></i>
636
</a>
<a class="diff-line-num" data-line-number="637" href="#L637" id="L637">
<i class="fa fa-link"></i>
637
</a>
<a class="diff-line-num" data-line-number="638" href="#L638" id="L638">
<i class="fa fa-link"></i>
638
</a>
<a class="diff-line-num" data-line-number="639" href="#L639" id="L639">
<i class="fa fa-link"></i>
639
</a>
<a class="diff-line-num" data-line-number="640" href="#L640" id="L640">
<i class="fa fa-link"></i>
640
</a>
<a class="diff-line-num" data-line-number="641" href="#L641" id="L641">
<i class="fa fa-link"></i>
641
</a>
<a class="diff-line-num" data-line-number="642" href="#L642" id="L642">
<i class="fa fa-link"></i>
642
</a>
<a class="diff-line-num" data-line-number="643" href="#L643" id="L643">
<i class="fa fa-link"></i>
643
</a>
<a class="diff-line-num" data-line-number="644" href="#L644" id="L644">
<i class="fa fa-link"></i>
644
</a>
<a class="diff-line-num" data-line-number="645" href="#L645" id="L645">
<i class="fa fa-link"></i>
645
</a>
<a class="diff-line-num" data-line-number="646" href="#L646" id="L646">
<i class="fa fa-link"></i>
646
</a>
<a class="diff-line-num" data-line-number="647" href="#L647" id="L647">
<i class="fa fa-link"></i>
647
</a>
<a class="diff-line-num" data-line-number="648" href="#L648" id="L648">
<i class="fa fa-link"></i>
648
</a>
<a class="diff-line-num" data-line-number="649" href="#L649" id="L649">
<i class="fa fa-link"></i>
649
</a>
<a class="diff-line-num" data-line-number="650" href="#L650" id="L650">
<i class="fa fa-link"></i>
650
</a>
<a class="diff-line-num" data-line-number="651" href="#L651" id="L651">
<i class="fa fa-link"></i>
651
</a>
<a class="diff-line-num" data-line-number="652" href="#L652" id="L652">
<i class="fa fa-link"></i>
652
</a>
<a class="diff-line-num" data-line-number="653" href="#L653" id="L653">
<i class="fa fa-link"></i>
653
</a>
<a class="diff-line-num" data-line-number="654" href="#L654" id="L654">
<i class="fa fa-link"></i>
654
</a>
<a class="diff-line-num" data-line-number="655" href="#L655" id="L655">
<i class="fa fa-link"></i>
655
</a>
<a class="diff-line-num" data-line-number="656" href="#L656" id="L656">
<i class="fa fa-link"></i>
656
</a>
<a class="diff-line-num" data-line-number="657" href="#L657" id="L657">
<i class="fa fa-link"></i>
657
</a>
<a class="diff-line-num" data-line-number="658" href="#L658" id="L658">
<i class="fa fa-link"></i>
658
</a>
<a class="diff-line-num" data-line-number="659" href="#L659" id="L659">
<i class="fa fa-link"></i>
659
</a>
<a class="diff-line-num" data-line-number="660" href="#L660" id="L660">
<i class="fa fa-link"></i>
660
</a>
<a class="diff-line-num" data-line-number="661" href="#L661" id="L661">
<i class="fa fa-link"></i>
661
</a>
<a class="diff-line-num" data-line-number="662" href="#L662" id="L662">
<i class="fa fa-link"></i>
662
</a>
<a class="diff-line-num" data-line-number="663" href="#L663" id="L663">
<i class="fa fa-link"></i>
663
</a>
<a class="diff-line-num" data-line-number="664" href="#L664" id="L664">
<i class="fa fa-link"></i>
664
</a>
<a class="diff-line-num" data-line-number="665" href="#L665" id="L665">
<i class="fa fa-link"></i>
665
</a>
<a class="diff-line-num" data-line-number="666" href="#L666" id="L666">
<i class="fa fa-link"></i>
666
</a>
<a class="diff-line-num" data-line-number="667" href="#L667" id="L667">
<i class="fa fa-link"></i>
667
</a>
<a class="diff-line-num" data-line-number="668" href="#L668" id="L668">
<i class="fa fa-link"></i>
668
</a>
<a class="diff-line-num" data-line-number="669" href="#L669" id="L669">
<i class="fa fa-link"></i>
669
</a>
<a class="diff-line-num" data-line-number="670" href="#L670" id="L670">
<i class="fa fa-link"></i>
670
</a>
<a class="diff-line-num" data-line-number="671" href="#L671" id="L671">
<i class="fa fa-link"></i>
671
</a>
<a class="diff-line-num" data-line-number="672" href="#L672" id="L672">
<i class="fa fa-link"></i>
672
</a>
<a class="diff-line-num" data-line-number="673" href="#L673" id="L673">
<i class="fa fa-link"></i>
673
</a>
<a class="diff-line-num" data-line-number="674" href="#L674" id="L674">
<i class="fa fa-link"></i>
674
</a>
<a class="diff-line-num" data-line-number="675" href="#L675" id="L675">
<i class="fa fa-link"></i>
675
</a>
<a class="diff-line-num" data-line-number="676" href="#L676" id="L676">
<i class="fa fa-link"></i>
676
</a>
<a class="diff-line-num" data-line-number="677" href="#L677" id="L677">
<i class="fa fa-link"></i>
677
</a>
<a class="diff-line-num" data-line-number="678" href="#L678" id="L678">
<i class="fa fa-link"></i>
678
</a>
<a class="diff-line-num" data-line-number="679" href="#L679" id="L679">
<i class="fa fa-link"></i>
679
</a>
<a class="diff-line-num" data-line-number="680" href="#L680" id="L680">
<i class="fa fa-link"></i>
680
</a>
<a class="diff-line-num" data-line-number="681" href="#L681" id="L681">
<i class="fa fa-link"></i>
681
</a>
<a class="diff-line-num" data-line-number="682" href="#L682" id="L682">
<i class="fa fa-link"></i>
682
</a>
<a class="diff-line-num" data-line-number="683" href="#L683" id="L683">
<i class="fa fa-link"></i>
683
</a>
<a class="diff-line-num" data-line-number="684" href="#L684" id="L684">
<i class="fa fa-link"></i>
684
</a>
<a class="diff-line-num" data-line-number="685" href="#L685" id="L685">
<i class="fa fa-link"></i>
685
</a>
<a class="diff-line-num" data-line-number="686" href="#L686" id="L686">
<i class="fa fa-link"></i>
686
</a>
<a class="diff-line-num" data-line-number="687" href="#L687" id="L687">
<i class="fa fa-link"></i>
687
</a>
<a class="diff-line-num" data-line-number="688" href="#L688" id="L688">
<i class="fa fa-link"></i>
688
</a>
<a class="diff-line-num" data-line-number="689" href="#L689" id="L689">
<i class="fa fa-link"></i>
689
</a>
<a class="diff-line-num" data-line-number="690" href="#L690" id="L690">
<i class="fa fa-link"></i>
690
</a>
<a class="diff-line-num" data-line-number="691" href="#L691" id="L691">
<i class="fa fa-link"></i>
691
</a>
<a class="diff-line-num" data-line-number="692" href="#L692" id="L692">
<i class="fa fa-link"></i>
692
</a>
<a class="diff-line-num" data-line-number="693" href="#L693" id="L693">
<i class="fa fa-link"></i>
693
</a>
<a class="diff-line-num" data-line-number="694" href="#L694" id="L694">
<i class="fa fa-link"></i>
694
</a>
<a class="diff-line-num" data-line-number="695" href="#L695" id="L695">
<i class="fa fa-link"></i>
695
</a>
<a class="diff-line-num" data-line-number="696" href="#L696" id="L696">
<i class="fa fa-link"></i>
696
</a>
<a class="diff-line-num" data-line-number="697" href="#L697" id="L697">
<i class="fa fa-link"></i>
697
</a>
<a class="diff-line-num" data-line-number="698" href="#L698" id="L698">
<i class="fa fa-link"></i>
698
</a>
<a class="diff-line-num" data-line-number="699" href="#L699" id="L699">
<i class="fa fa-link"></i>
699
</a>
<a class="diff-line-num" data-line-number="700" href="#L700" id="L700">
<i class="fa fa-link"></i>
700
</a>
<a class="diff-line-num" data-line-number="701" href="#L701" id="L701">
<i class="fa fa-link"></i>
701
</a>
<a class="diff-line-num" data-line-number="702" href="#L702" id="L702">
<i class="fa fa-link"></i>
702
</a>
<a class="diff-line-num" data-line-number="703" href="#L703" id="L703">
<i class="fa fa-link"></i>
703
</a>
<a class="diff-line-num" data-line-number="704" href="#L704" id="L704">
<i class="fa fa-link"></i>
704
</a>
<a class="diff-line-num" data-line-number="705" href="#L705" id="L705">
<i class="fa fa-link"></i>
705
</a>
<a class="diff-line-num" data-line-number="706" href="#L706" id="L706">
<i class="fa fa-link"></i>
706
</a>
<a class="diff-line-num" data-line-number="707" href="#L707" id="L707">
<i class="fa fa-link"></i>
707
</a>
<a class="diff-line-num" data-line-number="708" href="#L708" id="L708">
<i class="fa fa-link"></i>
708
</a>
<a class="diff-line-num" data-line-number="709" href="#L709" id="L709">
<i class="fa fa-link"></i>
709
</a>
<a class="diff-line-num" data-line-number="710" href="#L710" id="L710">
<i class="fa fa-link"></i>
710
</a>
<a class="diff-line-num" data-line-number="711" href="#L711" id="L711">
<i class="fa fa-link"></i>
711
</a>
<a class="diff-line-num" data-line-number="712" href="#L712" id="L712">
<i class="fa fa-link"></i>
712
</a>
<a class="diff-line-num" data-line-number="713" href="#L713" id="L713">
<i class="fa fa-link"></i>
713
</a>
<a class="diff-line-num" data-line-number="714" href="#L714" id="L714">
<i class="fa fa-link"></i>
714
</a>
<a class="diff-line-num" data-line-number="715" href="#L715" id="L715">
<i class="fa fa-link"></i>
715
</a>
<a class="diff-line-num" data-line-number="716" href="#L716" id="L716">
<i class="fa fa-link"></i>
716
</a>
<a class="diff-line-num" data-line-number="717" href="#L717" id="L717">
<i class="fa fa-link"></i>
717
</a>
<a class="diff-line-num" data-line-number="718" href="#L718" id="L718">
<i class="fa fa-link"></i>
718
</a>
<a class="diff-line-num" data-line-number="719" href="#L719" id="L719">
<i class="fa fa-link"></i>
719
</a>
<a class="diff-line-num" data-line-number="720" href="#L720" id="L720">
<i class="fa fa-link"></i>
720
</a>
<a class="diff-line-num" data-line-number="721" href="#L721" id="L721">
<i class="fa fa-link"></i>
721
</a>
<a class="diff-line-num" data-line-number="722" href="#L722" id="L722">
<i class="fa fa-link"></i>
722
</a>
<a class="diff-line-num" data-line-number="723" href="#L723" id="L723">
<i class="fa fa-link"></i>
723
</a>
<a class="diff-line-num" data-line-number="724" href="#L724" id="L724">
<i class="fa fa-link"></i>
724
</a>
<a class="diff-line-num" data-line-number="725" href="#L725" id="L725">
<i class="fa fa-link"></i>
725
</a>
<a class="diff-line-num" data-line-number="726" href="#L726" id="L726">
<i class="fa fa-link"></i>
726
</a>
<a class="diff-line-num" data-line-number="727" href="#L727" id="L727">
<i class="fa fa-link"></i>
727
</a>
<a class="diff-line-num" data-line-number="728" href="#L728" id="L728">
<i class="fa fa-link"></i>
728
</a>
<a class="diff-line-num" data-line-number="729" href="#L729" id="L729">
<i class="fa fa-link"></i>
729
</a>
<a class="diff-line-num" data-line-number="730" href="#L730" id="L730">
<i class="fa fa-link"></i>
730
</a>
<a class="diff-line-num" data-line-number="731" href="#L731" id="L731">
<i class="fa fa-link"></i>
731
</a>
<a class="diff-line-num" data-line-number="732" href="#L732" id="L732">
<i class="fa fa-link"></i>
732
</a>
<a class="diff-line-num" data-line-number="733" href="#L733" id="L733">
<i class="fa fa-link"></i>
733
</a>
<a class="diff-line-num" data-line-number="734" href="#L734" id="L734">
<i class="fa fa-link"></i>
734
</a>
<a class="diff-line-num" data-line-number="735" href="#L735" id="L735">
<i class="fa fa-link"></i>
735
</a>
<a class="diff-line-num" data-line-number="736" href="#L736" id="L736">
<i class="fa fa-link"></i>
736
</a>
<a class="diff-line-num" data-line-number="737" href="#L737" id="L737">
<i class="fa fa-link"></i>
737
</a>
</div>
<div class="blob-content" data-blob-id="bc1410b6238dd83672d40e8fff030117adcd2ed7">
<pre class="code highlight"><code><span id="LC1" class="line"><span class="c1">//</span></span>
<span id="LC2" class="line"><span class="c1">// Book:      OpenCL(R) Programming Guide</span></span>
<span id="LC3" class="line"><span class="c1">// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg</span></span>
<span id="LC4" class="line"><span class="c1">// ISBN-10:   0-321-74964-2</span></span>
<span id="LC5" class="line"><span class="c1">// ISBN-13:   978-0-321-74964-2</span></span>
<span id="LC6" class="line"><span class="c1">// Publisher: Addison-Wesley Professional</span></span>
<span id="LC7" class="line"><span class="c1">// URLs:      http://safari.informit.com/9780132488006/</span></span>
<span id="LC8" class="line"><span class="c1">//            http://www.openclprogrammingguide.com</span></span>
<span id="LC9" class="line"><span class="c1">//</span></span>
<span id="LC10" class="line"></span>
<span id="LC11" class="line"><span class="cp">#include &lt;windows.h&gt;</span></span>
<span id="LC12" class="line"><span class="cp">#include &lt;dxgi.h&gt;</span></span>
<span id="LC13" class="line"><span class="cp">#include &lt;d3d10.h&gt;</span></span>
<span id="LC14" class="line"><span class="cp">#include &lt;d3dx10.h&gt;</span></span>
<span id="LC15" class="line"><span class="cp">#include &lt;iostream&gt;</span></span>
<span id="LC16" class="line"><span class="cp">#include &lt;fstream&gt;</span></span>
<span id="LC17" class="line"><span class="cp">#include &lt;sstream&gt;</span></span>
<span id="LC18" class="line"><span class="cp">#include &lt;cassert&gt;</span></span>
<span id="LC19" class="line"><span class="cp">#include &lt;CL/cl.h&gt;</span></span>
<span id="LC20" class="line"><span class="cm">/* OpenCL D3D10 interop functions are available from the header "cl_d3d10.h".  </span></span>
<span id="LC21" class="line"><span class="cm">   Note that the Khronos extensions for D3D10 are available on the Khronos website.  </span></span>
<span id="LC22" class="line"><span class="cm">   On some distributions you may need to download this file.  </span></span>
<span id="LC23" class="line"><span class="cm">   The sample code assumes this is found in the OpenCL include path. */</span></span>
<span id="LC24" class="line"><span class="cp">#include &lt;CL/cl_d3d10.h&gt;</span></span>
<span id="LC25" class="line"><span class="cp">#include &lt;CL/cl_ext.h&gt;</span></span>
<span id="LC26" class="line"><span class="cp">#pragma OPENCL EXTENSION cl_khr_d3d10_sharing  : enable</span></span>
<span id="LC27" class="line"><span class="cp">#ifndef SAFE_RELEASE</span></span>
<span id="LC28" class="line"><span class="cp">#define SAFE_RELEASE(p)      { if (p) { (p)-&gt;Release(); (p)=NULL; } }</span></span>
<span id="LC29" class="line"><span class="cp">#endif</span></span>
<span id="LC30" class="line"><span class="n">clGetDeviceIDsFromD3D10KHR_fn</span>       <span class="n">clGetDeviceIDsFromD3D10KHR</span>      <span class="o">=</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC31" class="line"><span class="n">clCreateFromD3D10BufferKHR_fn</span>		<span class="n">clCreateFromD3D10BufferKHR</span>      <span class="o">=</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC32" class="line"><span class="n">clCreateFromD3D10Texture2DKHR_fn</span>	<span class="n">clCreateFromD3D10Texture2DKHR</span>   <span class="o">=</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC33" class="line"><span class="n">clCreateFromD3D10Texture3DKHR_fn</span>    <span class="n">clCreateFromD3D10Texture3DKHR</span>   <span class="o">=</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC34" class="line"><span class="n">clEnqueueAcquireD3D10ObjectsKHR_fn</span>	<span class="n">clEnqueueAcquireD3D10ObjectsKHR</span> <span class="o">=</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC35" class="line"><span class="n">clEnqueueReleaseD3D10ObjectsKHR_fn</span>	<span class="n">clEnqueueReleaseD3D10ObjectsKHR</span> <span class="o">=</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC36" class="line"><span class="cp">#define INITPFN(x) \</span></span>
<span id="LC37" class="line">    <span class="n">x</span> <span class="o">=</span> <span class="p">(</span><span class="n">x</span> <span class="err">##</span> <span class="n">_fn</span><span class="p">)</span><span class="n">clGetExtensionFunctionAddress</span><span class="p">(</span><span class="err">#</span><span class="n">x</span><span class="p">);</span><span class="err">\</span></span>
<span id="LC38" class="line">	<span class="k">if</span><span class="p">(</span><span class="o">!</span><span class="n">x</span><span class="p">)</span> <span class="p">{</span> <span class="n">printf</span><span class="p">(</span><span class="s">"failed getting %s"</span> <span class="err">#</span><span class="n">x</span><span class="p">);</span> <span class="p">}</span></span>
<span id="LC39" class="line"><span class="c1">//--------------------------------------------------------------------------------------</span></span>
<span id="LC40" class="line"><span class="c1">// Global Variables</span></span>
<span id="LC41" class="line"><span class="c1">//--------------------------------------------------------------------------------------</span></span>
<span id="LC42" class="line"><span class="n">HWND</span>        <span class="n">g_hWnd</span> <span class="o">=</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC43" class="line"></span>
<span id="LC44" class="line"><span class="k">const</span> <span class="kt">unsigned</span> <span class="kt">int</span>    <span class="n">g_WindowWidth</span>  <span class="o">=</span> <span class="mi">256</span><span class="p">;</span></span>
<span id="LC45" class="line"><span class="k">const</span> <span class="kt">unsigned</span> <span class="kt">int</span>    <span class="n">g_WindowHeight</span> <span class="o">=</span> <span class="mi">256</span><span class="p">;</span></span>
<span id="LC46" class="line"></span>
<span id="LC47" class="line"><span class="c1">// Global D3D device/context/feature pointers</span></span>
<span id="LC48" class="line"><span class="n">ID3D10Device</span> <span class="o">*</span><span class="n">g_pD3DDevice</span><span class="p">;</span></span>
<span id="LC49" class="line"><span class="n">IDXGISwapChain</span> <span class="o">*</span><span class="n">g_pSwapChain</span><span class="p">;</span></span>
<span id="LC50" class="line"><span class="n">D3D_FEATURE_LEVEL</span> <span class="n">g_D3DFeatureLevel</span><span class="p">;</span></span>
<span id="LC51" class="line"><span class="n">ID3D10RenderTargetView</span> <span class="o">*</span><span class="n">g_pRenderTargetView</span><span class="p">;</span></span>
<span id="LC52" class="line"></span>
<span id="LC53" class="line"><span class="c1">// stuff for rendering a triangle, the vertex shader, the vertex buffer, and the input layout structure.</span></span>
<span id="LC54" class="line"><span class="n">ID3D10Buffer</span><span class="o">*</span>		<span class="n">g_pVertexBuffer</span> <span class="o">=</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC55" class="line"><span class="n">ID3D10Buffer</span><span class="o">*</span>		<span class="n">g_pSineVertexBuffer</span> <span class="o">=</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC56" class="line"><span class="n">ID3D10InputLayout</span><span class="o">*</span>      <span class="n">g_pVertexLayout</span> <span class="o">=</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC57" class="line"><span class="n">ID3D10InputLayout</span><span class="o">*</span>		<span class="n">g_pSineVertexLayout</span> <span class="o">=</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC58" class="line"><span class="n">ID3D10Effect</span><span class="o">*</span>		<span class="n">g_pEffect</span> <span class="o">=</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC59" class="line"><span class="n">ID3D10Texture2D</span><span class="o">*</span>	<span class="n">g_pTexture2D</span> <span class="o">=</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC60" class="line"><span class="n">ID3D10EffectTechnique</span><span class="o">*</span>  <span class="n">g_pTechnique</span> <span class="o">=</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC61" class="line"><span class="n">ID3D10EffectShaderResourceVariable</span><span class="o">*</span> <span class="n">g_pDiffuseVariable</span> <span class="o">=</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC62" class="line"><span class="n">ID3D10ShaderResourceView</span> <span class="o">*</span><span class="n">pSRView</span> <span class="o">=</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC63" class="line"></span>
<span id="LC64" class="line"><span class="k">struct</span> <span class="n">SimpleVertex</span></span>
<span id="LC65" class="line"><span class="p">{</span></span>
<span id="LC66" class="line">    <span class="n">D3DXVECTOR3</span> <span class="n">Pos</span><span class="p">;</span></span>
<span id="LC67" class="line">    <span class="n">D3DXVECTOR2</span> <span class="n">Tex</span><span class="p">;</span> <span class="c1">// Texture Coordinate</span></span>
<span id="LC68" class="line"><span class="p">};</span></span>
<span id="LC69" class="line"></span>
<span id="LC70" class="line"><span class="k">struct</span> <span class="n">SimpleSineVertex</span></span>
<span id="LC71" class="line"><span class="p">{</span></span>
<span id="LC72" class="line">	<span class="n">D3DXVECTOR4</span> <span class="n">Pos</span><span class="p">;</span></span>
<span id="LC73" class="line"><span class="p">};</span></span>
<span id="LC74" class="line"></span>
<span id="LC75" class="line"></span>
<span id="LC76" class="line"><span class="kt">bool</span> <span class="n">verbose</span> <span class="o">=</span> <span class="nb">true</span><span class="p">;</span></span>
<span id="LC77" class="line"></span>
<span id="LC78" class="line"></span>
<span id="LC79" class="line"><span class="c1">// OpenCL global defines</span></span>
<span id="LC80" class="line"><span class="n">cl_command_queue</span> <span class="n">commandQueue</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC81" class="line"><span class="n">cl_program</span> <span class="n">program</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC82" class="line"><span class="n">cl_mem</span> <span class="n">g_clTexture2D</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC83" class="line"><span class="n">cl_mem</span> <span class="n">g_clBuffer</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC84" class="line"><span class="n">cl_kernel</span> <span class="n">tex_kernel</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC85" class="line"><span class="n">cl_kernel</span> <span class="n">buffer_kernel</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC86" class="line"><span class="n">cl_context</span> <span class="n">context</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC87" class="line"></span>
<span id="LC88" class="line"></span>
<span id="LC89" class="line"><span class="c1">//--------------------------------------------------------------------------------------</span></span>
<span id="LC90" class="line"><span class="c1">// Forward declarations</span></span>
<span id="LC91" class="line"><span class="c1">//--------------------------------------------------------------------------------------</span></span>
<span id="LC92" class="line"><span class="n">HRESULT</span> <span class="n">InitWindow</span><span class="p">(</span> <span class="n">HINSTANCE</span> <span class="n">hInstance</span><span class="p">,</span> <span class="kt">int</span> <span class="n">nCmdShow</span> <span class="p">);</span></span>
<span id="LC93" class="line"><span class="n">HRESULT</span> <span class="n">InitTextures</span><span class="p">(</span><span class="n">cl_context</span> <span class="n">context</span><span class="p">);</span></span>
<span id="LC94" class="line"><span class="n">HRESULT</span> <span class="n">createRenderTargetViewOfSwapChainBackBuffer</span><span class="p">(</span><span class="kt">int</span> <span class="n">width</span><span class="p">,</span> <span class="kt">int</span> <span class="n">height</span><span class="p">);</span></span>
<span id="LC95" class="line"><span class="n">HRESULT</span> <span class="n">InitDeviceAndSwapChain</span><span class="p">(</span><span class="kt">int</span> <span class="n">width</span><span class="p">,</span> <span class="kt">int</span> <span class="n">height</span><span class="p">);</span></span>
<span id="LC96" class="line"><span class="kt">void</span> <span class="n">Cleanup</span><span class="p">();</span></span>
<span id="LC97" class="line"></span>
<span id="LC98" class="line"><span class="c1">///</span></span>
<span id="LC99" class="line"><span class="c1">// Desc: Initializes Direct3D Textures (allocation and initialization)</span></span>
<span id="LC100" class="line"><span class="c1">//</span></span>
<span id="LC101" class="line"><span class="n">HRESULT</span> <span class="nf">InitTextures</span><span class="p">(</span><span class="n">cl_context</span> <span class="n">context</span><span class="p">)</span></span>
<span id="LC102" class="line"><span class="p">{</span></span>
<span id="LC103" class="line">	<span class="n">cl_int</span> <span class="n">errNum</span><span class="p">;</span></span>
<span id="LC104" class="line">	<span class="c1">//</span></span>
<span id="LC105" class="line">	<span class="c1">// create the D3D resources we'll be using</span></span>
<span id="LC106" class="line">	<span class="c1">//</span></span>
<span id="LC107" class="line">	<span class="c1">// 2D texture</span></span>
<span id="LC108" class="line">	<span class="n">D3D10_TEXTURE2D_DESC</span> <span class="n">desc</span><span class="p">;</span></span>
<span id="LC109" class="line">	<span class="n">ZeroMemory</span><span class="p">(</span> <span class="o">&amp;</span><span class="n">desc</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="n">D3D10_TEXTURE2D_DESC</span><span class="p">)</span> <span class="p">);</span></span>
<span id="LC110" class="line">	<span class="n">desc</span><span class="p">.</span><span class="n">Width</span> <span class="o">=</span> <span class="n">g_WindowWidth</span><span class="p">;</span></span>
<span id="LC111" class="line">	<span class="n">desc</span><span class="p">.</span><span class="n">Height</span> <span class="o">=</span> <span class="n">g_WindowHeight</span><span class="p">;</span></span>
<span id="LC112" class="line">	<span class="n">desc</span><span class="p">.</span><span class="n">MipLevels</span> <span class="o">=</span> <span class="mi">1</span><span class="p">;</span></span>
<span id="LC113" class="line">	<span class="n">desc</span><span class="p">.</span><span class="n">ArraySize</span> <span class="o">=</span> <span class="mi">1</span><span class="p">;</span></span>
<span id="LC114" class="line">	<span class="n">desc</span><span class="p">.</span><span class="n">Format</span> <span class="o">=</span> <span class="n">DXGI_FORMAT_R8G8B8A8_UNORM</span><span class="p">;</span></span>
<span id="LC115" class="line">	<span class="n">desc</span><span class="p">.</span><span class="n">SampleDesc</span><span class="p">.</span><span class="n">Count</span> <span class="o">=</span> <span class="mi">1</span><span class="p">;</span></span>
<span id="LC116" class="line">	<span class="n">desc</span><span class="p">.</span><span class="n">Usage</span> <span class="o">=</span> <span class="n">D3D10_USAGE_DEFAULT</span><span class="p">;</span></span>
<span id="LC117" class="line">	<span class="n">desc</span><span class="p">.</span><span class="n">BindFlags</span> <span class="o">=</span> <span class="n">D3D10_BIND_SHADER_RESOURCE</span><span class="p">;</span></span>
<span id="LC118" class="line">	<span class="k">if</span> <span class="p">(</span><span class="n">FAILED</span><span class="p">(</span><span class="n">g_pD3DDevice</span><span class="o">-&gt;</span><span class="n">CreateTexture2D</span><span class="p">(</span> <span class="o">&amp;</span><span class="n">desc</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">g_pTexture2D</span><span class="p">)))</span></span>
<span id="LC119" class="line">		<span class="k">return</span> <span class="n">E_FAIL</span><span class="p">;</span></span>
<span id="LC120" class="line"></span>
<span id="LC121" class="line">	<span class="k">if</span> <span class="p">(</span><span class="n">FAILED</span><span class="p">(</span><span class="n">g_pD3DDevice</span><span class="o">-&gt;</span><span class="n">CreateShaderResourceView</span><span class="p">(</span><span class="n">g_pTexture2D</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">pSRView</span><span class="p">))</span> <span class="p">)</span></span>
<span id="LC122" class="line">		<span class="k">return</span> <span class="n">E_FAIL</span><span class="p">;</span></span>
<span id="LC123" class="line">	<span class="n">g_pDiffuseVariable</span><span class="o">-&gt;</span><span class="n">SetResource</span><span class="p">(</span><span class="n">pSRView</span> <span class="p">);</span></span>
<span id="LC124" class="line">	<span class="c1">// Create the OpenCL part</span></span>
<span id="LC125" class="line">	<span class="n">g_clTexture2D</span> <span class="o">=</span> <span class="n">clCreateFromD3D10Texture2DKHR</span><span class="p">(</span></span>
<span id="LC126" class="line">		<span class="n">context</span><span class="p">,</span></span>
<span id="LC127" class="line">		<span class="n">CL_MEM_READ_WRITE</span><span class="p">,</span></span>
<span id="LC128" class="line">		<span class="n">g_pTexture2D</span><span class="p">,</span></span>
<span id="LC129" class="line">		<span class="mi">0</span><span class="p">,</span></span>
<span id="LC130" class="line">		<span class="o">&amp;</span><span class="n">errNum</span><span class="p">);</span></span>
<span id="LC131" class="line">	<span class="k">if</span> <span class="p">(</span><span class="n">errNum</span> <span class="o">!=</span> <span class="n">CL_SUCCESS</span><span class="p">)</span></span>
<span id="LC132" class="line">	<span class="p">{</span></span>
<span id="LC133" class="line">		<span class="k">if</span><span class="p">(</span> <span class="n">errNum</span> <span class="o">==</span> <span class="n">CL_INVALID_D3D10_RESOURCE_KHR</span> <span class="p">)</span> <span class="p">{</span></span>
<span id="LC134" class="line">			<span class="n">std</span><span class="o">::</span><span class="n">cerr</span><span class="o">&lt;&lt;</span><span class="s">"Invalid d3d10 texture resource"</span><span class="o">&lt;&lt;</span><span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC135" class="line">		<span class="p">}</span></span>
<span id="LC136" class="line">		<span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Error creating 2D CL texture from D3D10"</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC137" class="line">		<span class="k">return</span> <span class="n">E_FAIL</span><span class="p">;</span></span>
<span id="LC138" class="line">	<span class="p">}</span></span>
<span id="LC139" class="line"></span>
<span id="LC140" class="line">	<span class="n">g_clBuffer</span> <span class="o">=</span> <span class="n">clCreateFromD3D10BufferKHR</span><span class="p">(</span> <span class="n">context</span><span class="p">,</span> <span class="n">CL_MEM_READ_WRITE</span><span class="p">,</span> <span class="n">g_pSineVertexBuffer</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">errNum</span> <span class="p">);</span></span>
<span id="LC141" class="line">	<span class="k">if</span><span class="p">(</span> <span class="n">errNum</span> <span class="o">!=</span> <span class="n">CL_SUCCESS</span><span class="p">)</span></span>
<span id="LC142" class="line">	<span class="p">{</span></span>
<span id="LC143" class="line"></span>
<span id="LC144" class="line">		<span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Error creating buffer from D3D10"</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC145" class="line">		<span class="k">return</span> <span class="n">E_FAIL</span><span class="p">;</span></span>
<span id="LC146" class="line">	<span class="p">}</span></span>
<span id="LC147" class="line"></span>
<span id="LC148" class="line">	<span class="k">return</span> <span class="n">S_OK</span><span class="p">;</span></span>
<span id="LC149" class="line"><span class="p">}</span></span>
<span id="LC150" class="line"></span>
<span id="LC151" class="line"></span>
<span id="LC152" class="line"></span>
<span id="LC153" class="line"><span class="c1">///</span></span>
<span id="LC154" class="line"><span class="c1">// Use OpenCL to compute the colors on the texture background</span></span>
<span id="LC155" class="line"><span class="n">cl_int</span> <span class="nf">computeTexture</span><span class="p">()</span></span>
<span id="LC156" class="line"><span class="p">{</span></span>
<span id="LC157" class="line">	<span class="n">cl_int</span> <span class="n">errNum</span><span class="p">;</span></span>
<span id="LC158" class="line"></span>
<span id="LC159" class="line">	<span class="k">static</span> <span class="n">cl_int</span> <span class="n">seq</span> <span class="o">=</span><span class="mi">0</span><span class="p">;</span></span>
<span id="LC160" class="line">	<span class="n">seq</span> <span class="o">=</span> <span class="p">(</span><span class="n">seq</span><span class="o">+</span><span class="mi">1</span><span class="p">)</span><span class="o">%</span><span class="p">(</span><span class="n">g_WindowWidth</span><span class="o">*</span><span class="mi">2</span><span class="p">);</span></span>
<span id="LC161" class="line"></span>
<span id="LC162" class="line">    <span class="n">errNum</span> <span class="o">=</span> <span class="n">clSetKernelArg</span><span class="p">(</span><span class="n">tex_kernel</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="n">cl_mem</span><span class="p">),</span> <span class="o">&amp;</span><span class="n">g_clTexture2D</span><span class="p">);</span></span>
<span id="LC163" class="line">    <span class="n">errNum</span> <span class="o">=</span> <span class="n">clSetKernelArg</span><span class="p">(</span><span class="n">tex_kernel</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="n">cl_int</span><span class="p">),</span> <span class="o">&amp;</span><span class="n">g_WindowWidth</span><span class="p">);</span></span>
<span id="LC164" class="line">    <span class="n">errNum</span> <span class="o">=</span> <span class="n">clSetKernelArg</span><span class="p">(</span><span class="n">tex_kernel</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="n">cl_int</span><span class="p">),</span> <span class="o">&amp;</span><span class="n">g_WindowHeight</span><span class="p">);</span></span>
<span id="LC165" class="line">    <span class="n">errNum</span> <span class="o">=</span> <span class="n">clSetKernelArg</span><span class="p">(</span><span class="n">tex_kernel</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="n">cl_int</span><span class="p">),</span> <span class="o">&amp;</span><span class="n">seq</span><span class="p">);</span></span>
<span id="LC166" class="line">	</span>
<span id="LC167" class="line">	<span class="kt">size_t</span> <span class="n">tex_globalWorkSize</span><span class="p">[</span><span class="mi">2</span><span class="p">]</span> <span class="o">=</span> <span class="p">{</span> <span class="n">g_WindowWidth</span><span class="p">,</span> <span class="n">g_WindowHeight</span> <span class="p">};</span></span>
<span id="LC168" class="line">	<span class="kt">size_t</span> <span class="n">tex_localWorkSize</span><span class="p">[</span><span class="mi">2</span><span class="p">]</span> <span class="o">=</span> <span class="p">{</span> <span class="mi">32</span><span class="p">,</span> <span class="mi">4</span> <span class="p">}</span> <span class="p">;</span></span>
<span id="LC169" class="line"></span>
<span id="LC170" class="line">	<span class="n">errNum</span> <span class="o">=</span> <span class="n">clEnqueueAcquireD3D10ObjectsKHR</span><span class="p">(</span><span class="n">commandQueue</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">g_clTexture2D</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span> <span class="p">);</span></span>
<span id="LC171" class="line"></span>
<span id="LC172" class="line">    <span class="n">errNum</span> <span class="o">=</span> <span class="n">clEnqueueNDRangeKernel</span><span class="p">(</span><span class="n">commandQueue</span><span class="p">,</span> <span class="n">tex_kernel</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span></span>
<span id="LC173" class="line">                                    <span class="n">tex_globalWorkSize</span><span class="p">,</span> <span class="n">tex_localWorkSize</span><span class="p">,</span></span>
<span id="LC174" class="line">                                    <span class="mi">0</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">);</span></span>
<span id="LC175" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">errNum</span> <span class="o">!=</span> <span class="n">CL_SUCCESS</span><span class="p">)</span></span>
<span id="LC176" class="line">    <span class="p">{</span></span>
<span id="LC177" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Error queuing kernel for execution."</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC178" class="line">    <span class="p">}</span></span>
<span id="LC179" class="line">	<span class="n">errNum</span> <span class="o">=</span> <span class="n">clEnqueueReleaseD3D10ObjectsKHR</span><span class="p">(</span><span class="n">commandQueue</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">g_clTexture2D</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span> <span class="p">);</span></span>
<span id="LC180" class="line">	<span class="n">clFinish</span><span class="p">(</span><span class="n">commandQueue</span><span class="p">);</span></span>
<span id="LC181" class="line">	<span class="k">return</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC182" class="line"><span class="p">}</span></span>
<span id="LC183" class="line"></span>
<span id="LC184" class="line"><span class="c1">///</span></span>
<span id="LC185" class="line"><span class="c1">// Use OpenCL to compute the colors on the texture background</span></span>
<span id="LC186" class="line"><span class="n">cl_int</span> <span class="nf">computeBuffer</span><span class="p">()</span></span>
<span id="LC187" class="line"><span class="p">{</span></span>
<span id="LC188" class="line">	<span class="n">cl_int</span> <span class="n">errNum</span><span class="p">;</span></span>
<span id="LC189" class="line"></span>
<span id="LC190" class="line">	<span class="k">static</span> <span class="n">cl_int</span> <span class="n">seq</span> <span class="o">=</span><span class="mi">0</span><span class="p">;</span></span>
<span id="LC191" class="line">	<span class="n">seq</span> <span class="o">=</span> <span class="p">(</span><span class="n">seq</span><span class="o">+</span><span class="mi">1</span><span class="p">)</span><span class="o">%</span><span class="p">(</span><span class="n">g_WindowWidth</span><span class="o">*</span><span class="mi">2</span><span class="p">);</span></span>
<span id="LC192" class="line"></span>
<span id="LC193" class="line">    <span class="n">errNum</span> <span class="o">=</span> <span class="n">clSetKernelArg</span><span class="p">(</span><span class="n">buffer_kernel</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="n">cl_mem</span><span class="p">),</span> <span class="o">&amp;</span><span class="n">g_clBuffer</span><span class="p">);</span></span>
<span id="LC194" class="line">    <span class="n">errNum</span> <span class="o">=</span> <span class="n">clSetKernelArg</span><span class="p">(</span><span class="n">buffer_kernel</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="n">cl_int</span><span class="p">),</span> <span class="o">&amp;</span><span class="n">g_WindowWidth</span><span class="p">);</span></span>
<span id="LC195" class="line">    <span class="n">errNum</span> <span class="o">=</span> <span class="n">clSetKernelArg</span><span class="p">(</span><span class="n">buffer_kernel</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="n">cl_int</span><span class="p">),</span> <span class="o">&amp;</span><span class="n">g_WindowHeight</span><span class="p">);</span></span>
<span id="LC196" class="line">    <span class="n">errNum</span> <span class="o">=</span> <span class="n">clSetKernelArg</span><span class="p">(</span><span class="n">buffer_kernel</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span><span class="n">cl_int</span><span class="p">),</span> <span class="o">&amp;</span><span class="n">seq</span><span class="p">);</span></span>
<span id="LC197" class="line">	</span>
<span id="LC198" class="line">	<span class="kt">size_t</span> <span class="n">buffer_globalWorkSize</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span> <span class="o">=</span> <span class="p">{</span> <span class="n">g_WindowWidth</span> <span class="p">};</span></span>
<span id="LC199" class="line">	<span class="kt">size_t</span> <span class="n">buffer_localWorkSize</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span> <span class="o">=</span> <span class="p">{</span> <span class="mi">32</span> <span class="p">}</span> <span class="p">;</span></span>
<span id="LC200" class="line"></span>
<span id="LC201" class="line">	<span class="n">errNum</span> <span class="o">=</span> <span class="n">clEnqueueAcquireD3D10ObjectsKHR</span><span class="p">(</span><span class="n">commandQueue</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">g_clBuffer</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span> <span class="p">);</span></span>
<span id="LC202" class="line"></span>
<span id="LC203" class="line">    <span class="n">errNum</span> <span class="o">=</span> <span class="n">clEnqueueNDRangeKernel</span><span class="p">(</span><span class="n">commandQueue</span><span class="p">,</span> <span class="n">buffer_kernel</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span></span>
<span id="LC204" class="line">                                    <span class="n">buffer_globalWorkSize</span><span class="p">,</span> <span class="n">buffer_localWorkSize</span><span class="p">,</span></span>
<span id="LC205" class="line">                                    <span class="mi">0</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">);</span></span>
<span id="LC206" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">errNum</span> <span class="o">!=</span> <span class="n">CL_SUCCESS</span><span class="p">)</span></span>
<span id="LC207" class="line">    <span class="p">{</span></span>
<span id="LC208" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Error queuing kernel for execution."</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC209" class="line">    <span class="p">}</span></span>
<span id="LC210" class="line">	<span class="n">errNum</span> <span class="o">=</span> <span class="n">clEnqueueReleaseD3D10ObjectsKHR</span><span class="p">(</span><span class="n">commandQueue</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">g_clBuffer</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span> <span class="p">);</span></span>
<span id="LC211" class="line">	<span class="n">clFinish</span><span class="p">(</span><span class="n">commandQueue</span><span class="p">);</span></span>
<span id="LC212" class="line">	<span class="k">return</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC213" class="line"><span class="p">}</span></span>
<span id="LC214" class="line"></span>
<span id="LC215" class="line"></span>
<span id="LC216" class="line"><span class="c1">//--------------------------------------------------------------------------------------</span></span>
<span id="LC217" class="line"><span class="c1">// Render a frame</span></span>
<span id="LC218" class="line"><span class="c1">//--------------------------------------------------------------------------------------</span></span>
<span id="LC219" class="line"><span class="kt">void</span> <span class="nf">Render</span><span class="p">()</span></span>
<span id="LC220" class="line"><span class="p">{</span></span>
<span id="LC221" class="line">    <span class="c1">// Clear the back buffer </span></span>
<span id="LC222" class="line">    <span class="kt">float</span> <span class="n">ClearColor</span><span class="p">[</span><span class="mi">4</span><span class="p">]</span> <span class="o">=</span> <span class="p">{</span> <span class="mf">0.0</span><span class="n">f</span><span class="p">,</span> <span class="mf">0.125</span><span class="n">f</span><span class="p">,</span> <span class="mf">0.1</span><span class="n">f</span><span class="p">,</span> <span class="mf">1.0</span><span class="n">f</span> <span class="p">};</span> <span class="c1">// red,green,blue,alpha</span></span>
<span id="LC223" class="line">	<span class="n">g_pD3DDevice</span><span class="o">-&gt;</span><span class="n">ClearRenderTargetView</span><span class="p">(</span> <span class="n">g_pRenderTargetView</span><span class="p">,</span> <span class="n">ClearColor</span><span class="p">);</span></span>
<span id="LC224" class="line">    <span class="c1">// Set the input layout</span></span>
<span id="LC225" class="line">    <span class="n">g_pD3DDevice</span><span class="o">-&gt;</span><span class="n">IASetInputLayout</span><span class="p">(</span> <span class="n">g_pVertexLayout</span> <span class="p">);</span></span>
<span id="LC226" class="line">    <span class="c1">// Set vertex buffer</span></span>
<span id="LC227" class="line">    <span class="n">UINT</span> <span class="n">stride</span> <span class="o">=</span> <span class="k">sizeof</span><span class="p">(</span> <span class="n">SimpleVertex</span> <span class="p">);</span></span>
<span id="LC228" class="line">    <span class="n">UINT</span> <span class="n">offset</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC229" class="line">    <span class="n">g_pD3DDevice</span><span class="o">-&gt;</span><span class="n">IASetVertexBuffers</span><span class="p">(</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">g_pVertexBuffer</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">stride</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">offset</span> <span class="p">);</span></span>
<span id="LC230" class="line"></span>
<span id="LC231" class="line">    <span class="c1">// Set primitive topology</span></span>
<span id="LC232" class="line">    <span class="c1">//g_pD3DDevice-&gt;IASetPrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST );</span></span>
<span id="LC233" class="line">    <span class="n">g_pD3DDevice</span><span class="o">-&gt;</span><span class="n">IASetPrimitiveTopology</span><span class="p">(</span> <span class="n">D3D10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP</span> <span class="p">);</span></span>
<span id="LC234" class="line">	<span class="c1">//g_pDiffuseVariable = </span></span>
<span id="LC235" class="line">	<span class="c1">//	g_pEffect-&gt;GetVariableByName("txDiffuse")-&gt;AsShaderResource();</span></span>
<span id="LC236" class="line">	<span class="n">computeTexture</span><span class="p">();</span></span>
<span id="LC237" class="line"></span>
<span id="LC238" class="line"></span>
<span id="LC239" class="line">    <span class="c1">// Render the quadrilateral</span></span>
<span id="LC240" class="line">    <span class="n">D3D10_TECHNIQUE_DESC</span> <span class="n">techDesc</span><span class="p">;</span></span>
<span id="LC241" class="line">    <span class="n">g_pTechnique</span><span class="o">-&gt;</span><span class="n">GetDesc</span><span class="p">(</span> <span class="o">&amp;</span><span class="n">techDesc</span> <span class="p">);</span></span>
<span id="LC242" class="line">  <span class="c1">//  for( UINT p = 0; p &lt; techDesc.Passes; ++p )</span></span>
<span id="LC243" class="line"> <span class="c1">//   {</span></span>
<span id="LC244" class="line">        <span class="n">g_pTechnique</span><span class="o">-&gt;</span><span class="n">GetPassByIndex</span><span class="p">(</span> <span class="mi">0</span> <span class="p">)</span><span class="o">-&gt;</span><span class="n">Apply</span><span class="p">(</span> <span class="mi">0</span> <span class="p">);</span></span>
<span id="LC245" class="line">        <span class="n">g_pD3DDevice</span><span class="o">-&gt;</span><span class="n">Draw</span><span class="p">(</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">0</span> <span class="p">);</span></span>
<span id="LC246" class="line">  <span class="c1">//  }</span></span>
<span id="LC247" class="line"></span>
<span id="LC248" class="line">    <span class="c1">// Set the input layout</span></span>
<span id="LC249" class="line">    <span class="n">g_pD3DDevice</span><span class="o">-&gt;</span><span class="n">IASetInputLayout</span><span class="p">(</span> <span class="n">g_pSineVertexLayout</span> <span class="p">);</span></span>
<span id="LC250" class="line">    <span class="c1">// Set vertex buffer</span></span>
<span id="LC251" class="line">    <span class="n">stride</span> <span class="o">=</span> <span class="k">sizeof</span><span class="p">(</span> <span class="n">SimpleSineVertex</span> <span class="p">);</span></span>
<span id="LC252" class="line">    <span class="n">offset</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC253" class="line">    <span class="n">g_pD3DDevice</span><span class="o">-&gt;</span><span class="n">IASetVertexBuffers</span><span class="p">(</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">g_pSineVertexBuffer</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">stride</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">offset</span> <span class="p">);</span></span>
<span id="LC254" class="line"></span>
<span id="LC255" class="line">    <span class="c1">// Set primitive topology</span></span>
<span id="LC256" class="line">    <span class="n">g_pD3DDevice</span><span class="o">-&gt;</span><span class="n">IASetPrimitiveTopology</span><span class="p">(</span> <span class="n">D3D10_PRIMITIVE_TOPOLOGY_LINESTRIP</span> <span class="p">);</span></span>
<span id="LC257" class="line"></span>
<span id="LC258" class="line">	<span class="n">computeBuffer</span><span class="p">();</span></span>
<span id="LC259" class="line">        <span class="n">g_pTechnique</span><span class="o">-&gt;</span><span class="n">GetPassByIndex</span><span class="p">(</span> <span class="mi">1</span> <span class="p">)</span><span class="o">-&gt;</span><span class="n">Apply</span><span class="p">(</span> <span class="mi">0</span> <span class="p">);</span></span>
<span id="LC260" class="line">        <span class="n">g_pD3DDevice</span><span class="o">-&gt;</span><span class="n">Draw</span><span class="p">(</span> <span class="mi">256</span><span class="p">,</span> <span class="mi">0</span> <span class="p">);</span></span>
<span id="LC261" class="line"></span>
<span id="LC262" class="line"></span>
<span id="LC263" class="line">    <span class="c1">// Present the information rendered to the back buffer to the front buffer (the screen)</span></span>
<span id="LC264" class="line">    <span class="n">g_pSwapChain</span><span class="o">-&gt;</span><span class="n">Present</span><span class="p">(</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span> <span class="p">);</span></span>
<span id="LC265" class="line"><span class="p">}</span></span>
<span id="LC266" class="line"><span class="c1">///</span></span>
<span id="LC267" class="line"><span class="c1">//  Create an OpenCL program from the kernel source file</span></span>
<span id="LC268" class="line"><span class="c1">//</span></span>
<span id="LC269" class="line"><span class="n">cl_program</span> <span class="nf">CreateProgram</span><span class="p">(</span><span class="n">cl_context</span> <span class="n">context</span><span class="p">,</span> <span class="n">cl_device_id</span> <span class="n">device</span><span class="p">,</span> <span class="k">const</span> <span class="kt">char</span><span class="o">*</span> <span class="n">fileName</span><span class="p">)</span></span>
<span id="LC270" class="line"><span class="p">{</span></span>
<span id="LC271" class="line">    <span class="n">cl_int</span> <span class="n">errNum</span><span class="p">;</span></span>
<span id="LC272" class="line">    <span class="n">cl_program</span> <span class="n">program</span><span class="p">;</span></span>
<span id="LC273" class="line"></span>
<span id="LC274" class="line">    <span class="n">std</span><span class="o">::</span><span class="n">ifstream</span> <span class="n">kernelFile</span><span class="p">(</span><span class="n">fileName</span><span class="p">,</span> <span class="n">std</span><span class="o">::</span><span class="n">ios</span><span class="o">::</span><span class="n">in</span><span class="p">);</span></span>
<span id="LC275" class="line">    <span class="k">if</span> <span class="p">(</span><span class="o">!</span><span class="n">kernelFile</span><span class="p">.</span><span class="n">is_open</span><span class="p">())</span></span>
<span id="LC276" class="line">    <span class="p">{</span></span>
<span id="LC277" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Failed to open file for reading: "</span> <span class="o">&lt;&lt;</span> <span class="n">fileName</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC278" class="line">        <span class="k">return</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC279" class="line">    <span class="p">}</span></span>
<span id="LC280" class="line"></span>
<span id="LC281" class="line">    <span class="n">std</span><span class="o">::</span><span class="n">ostringstream</span> <span class="n">oss</span><span class="p">;</span></span>
<span id="LC282" class="line">    <span class="n">oss</span> <span class="o">&lt;&lt;</span> <span class="n">kernelFile</span><span class="p">.</span><span class="n">rdbuf</span><span class="p">();</span></span>
<span id="LC283" class="line"></span>
<span id="LC284" class="line">    <span class="n">std</span><span class="o">::</span><span class="n">string</span> <span class="n">srcStdStr</span> <span class="o">=</span> <span class="n">oss</span><span class="p">.</span><span class="n">str</span><span class="p">();</span></span>
<span id="LC285" class="line">    <span class="k">const</span> <span class="kt">char</span> <span class="o">*</span><span class="n">srcStr</span> <span class="o">=</span> <span class="n">srcStdStr</span><span class="p">.</span><span class="n">c_str</span><span class="p">();</span></span>
<span id="LC286" class="line">    <span class="n">program</span> <span class="o">=</span> <span class="n">clCreateProgramWithSource</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span></span>
<span id="LC287" class="line">                                        <span class="p">(</span><span class="k">const</span> <span class="kt">char</span><span class="o">**</span><span class="p">)</span><span class="o">&amp;</span><span class="n">srcStr</span><span class="p">,</span></span>
<span id="LC288" class="line">                                        <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">);</span></span>
<span id="LC289" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">program</span> <span class="o">==</span> <span class="nb">NULL</span><span class="p">)</span></span>
<span id="LC290" class="line">    <span class="p">{</span></span>
<span id="LC291" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Failed to create CL program from source."</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC292" class="line">        <span class="k">return</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC293" class="line">    <span class="p">}</span></span>
<span id="LC294" class="line"></span>
<span id="LC295" class="line">    <span class="n">errNum</span> <span class="o">=</span> <span class="n">clBuildProgram</span><span class="p">(</span><span class="n">program</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">);</span></span>
<span id="LC296" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">errNum</span> <span class="o">!=</span> <span class="n">CL_SUCCESS</span><span class="p">)</span></span>
<span id="LC297" class="line">    <span class="p">{</span></span>
<span id="LC298" class="line">        <span class="c1">// Determine the reason for the error</span></span>
<span id="LC299" class="line">        <span class="kt">char</span> <span class="n">buildLog</span><span class="p">[</span><span class="mi">16384</span><span class="p">];</span></span>
<span id="LC300" class="line">        <span class="n">clGetProgramBuildInfo</span><span class="p">(</span><span class="n">program</span><span class="p">,</span> <span class="n">device</span><span class="p">,</span> <span class="n">CL_PROGRAM_BUILD_LOG</span><span class="p">,</span></span>
<span id="LC301" class="line">                              <span class="k">sizeof</span><span class="p">(</span><span class="n">buildLog</span><span class="p">),</span> <span class="n">buildLog</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">);</span></span>
<span id="LC302" class="line"></span>
<span id="LC303" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Error in kernel: "</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC304" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="n">buildLog</span><span class="p">;</span></span>
<span id="LC305" class="line">        <span class="n">clReleaseProgram</span><span class="p">(</span><span class="n">program</span><span class="p">);</span></span>
<span id="LC306" class="line">        <span class="k">return</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC307" class="line">    <span class="p">}</span></span>
<span id="LC308" class="line"></span>
<span id="LC309" class="line">    <span class="k">return</span> <span class="n">program</span><span class="p">;</span></span>
<span id="LC310" class="line"><span class="p">}</span></span>
<span id="LC311" class="line"></span>
<span id="LC312" class="line"><span class="kt">int</span> <span class="nf">main</span><span class="p">(</span> <span class="kt">int</span> <span class="n">argc</span><span class="p">,</span> <span class="k">const</span> <span class="kt">char</span><span class="o">**</span> <span class="n">argv</span><span class="p">[]</span> <span class="p">)</span></span>
<span id="LC313" class="line"><span class="p">{</span></span>
<span id="LC314" class="line">    <span class="n">cl_platform_id</span>	<span class="n">cpPlatform</span><span class="p">;</span></span>
<span id="LC315" class="line">	<span class="n">cl_int</span> <span class="n">errNum</span><span class="p">;</span></span>
<span id="LC316" class="line">	<span class="n">cl_uint</span> <span class="n">num_devices</span><span class="p">;</span></span>
<span id="LC317" class="line">	<span class="n">cl_device_id</span> <span class="n">cdDevice</span><span class="p">;</span></span>
<span id="LC318" class="line">    <span class="n">cl_uint</span> <span class="n">numPlatforms</span><span class="p">;</span></span>
<span id="LC319" class="line"></span>
<span id="LC320" class="line">	<span class="c1">//</span></span>
<span id="LC321" class="line">	<span class="c1">// Initialization of a D3D program contains the following steps:</span></span>
<span id="LC322" class="line">	<span class="c1">// 1. Initialize a Window, gets you hWnd (handle to the window)</span></span>
<span id="LC323" class="line">	<span class="c1">// 2. Initialize a SwapChain and Device, which can be done in a single call, makes a device points</span></span>
<span id="LC324" class="line">	<span class="c1">// 3. Create a RenderTargetView of the SwapChains BackBuffer.  </span></span>
<span id="LC325" class="line">	<span class="c1">//     a. Setup the viewport &amp; shader effects to pass through</span></span>
<span id="LC326" class="line">	<span class="c1">//</span></span>
<span id="LC327" class="line">    <span class="k">if</span><span class="p">(</span> <span class="n">FAILED</span><span class="p">(</span> <span class="n">InitWindow</span><span class="p">(</span> <span class="n">GetModuleHandle</span><span class="p">(</span><span class="nb">NULL</span><span class="p">),</span> <span class="n">SW_SHOWDEFAULT</span> <span class="p">)</span> <span class="p">)</span> <span class="p">)</span></span>
<span id="LC328" class="line">        <span class="k">return</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC329" class="line">	<span class="k">if</span><span class="p">(</span> <span class="n">FAILED</span><span class="p">(</span> <span class="n">InitDeviceAndSwapChain</span><span class="p">(</span> <span class="mi">640</span><span class="p">,</span> <span class="mi">480</span> <span class="p">)</span> <span class="p">)</span> <span class="p">)</span> </span>
<span id="LC330" class="line">		<span class="k">return</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC331" class="line">	<span class="k">if</span><span class="p">(</span> <span class="n">FAILED</span><span class="p">(</span> <span class="n">createRenderTargetViewOfSwapChainBackBuffer</span><span class="p">(</span><span class="mi">640</span><span class="p">,</span> <span class="mi">480</span><span class="p">)</span> <span class="p">)</span> <span class="p">)</span> </span>
<span id="LC332" class="line">		<span class="k">return</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC333" class="line"></span>
<span id="LC334" class="line">    <span class="c1">// First, select an OpenCL platform to run on.  For this example, we</span></span>
<span id="LC335" class="line">    <span class="c1">// simply choose the first available platform.  Normally, you would</span></span>
<span id="LC336" class="line">    <span class="c1">// query for all available platforms and select the most appropriate one.</span></span>
<span id="LC337" class="line">    <span class="n">errNum</span> <span class="o">=</span> <span class="n">clGetPlatformIDs</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">cpPlatform</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">numPlatforms</span><span class="p">);</span></span>
<span id="LC338" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">errNum</span> <span class="o">!=</span> <span class="n">CL_SUCCESS</span> <span class="o">||</span> <span class="n">numPlatforms</span> <span class="o">&lt;=</span> <span class="mi">0</span><span class="p">)</span></span>
<span id="LC339" class="line">    <span class="p">{</span></span>
<span id="LC340" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Failed to find any OpenCL platforms."</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC341" class="line">        <span class="k">return</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC342" class="line">    <span class="p">}</span></span>
<span id="LC343" class="line">	<span class="kt">char</span> <span class="n">extensionString</span><span class="p">[</span><span class="mi">256</span><span class="p">];</span></span>
<span id="LC344" class="line">	<span class="kt">size_t</span> <span class="n">extensionSize</span><span class="p">;</span></span>
<span id="LC345" class="line">	<span class="n">errNum</span> <span class="o">=</span> <span class="n">clGetPlatformInfo</span><span class="p">(</span> <span class="n">cpPlatform</span><span class="p">,</span> <span class="n">CL_PLATFORM_EXTENSIONS</span><span class="p">,</span> <span class="mi">256</span><span class="p">,</span> <span class="n">extensionString</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">extensionSize</span> <span class="p">);</span></span>
<span id="LC346" class="line"></span>
<span id="LC347" class="line">	<span class="c1">// We could parse the returned string to check for the availability of d3d_sharing extensions.</span></span>
<span id="LC348" class="line">	<span class="c1">// Here, we simply print it for code brevity, and assume it is present</span></span>
<span id="LC349" class="line">	<span class="n">std</span><span class="o">::</span><span class="n">cout</span><span class="o">&lt;&lt;</span><span class="s">"Extensions:</span><span class="se">\n\t</span><span class="s">"</span><span class="o">&lt;&lt;</span><span class="n">extensionString</span><span class="o">&lt;&lt;</span><span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC350" class="line"></span>
<span id="LC351" class="line">    <span class="c1">//</span></span>
<span id="LC352" class="line">	<span class="c1">// Initialize extension functions for D3D10</span></span>
<span id="LC353" class="line">	<span class="c1">// See the clGetExtensionFunctionAddress() documentation</span></span>
<span id="LC354" class="line">	<span class="c1">// for more details </span></span>
<span id="LC355" class="line">	<span class="c1">//</span></span>
<span id="LC356" class="line">	<span class="n">INITPFN</span><span class="p">(</span><span class="n">clGetDeviceIDsFromD3D10KHR</span><span class="p">);</span></span>
<span id="LC357" class="line">	<span class="n">INITPFN</span><span class="p">(</span><span class="n">clCreateFromD3D10BufferKHR</span><span class="p">);</span></span>
<span id="LC358" class="line">	<span class="n">INITPFN</span><span class="p">(</span><span class="n">clCreateFromD3D10Texture2DKHR</span><span class="p">);</span></span>
<span id="LC359" class="line">	<span class="n">INITPFN</span><span class="p">(</span><span class="n">clCreateFromD3D10Texture3DKHR</span><span class="p">);</span></span>
<span id="LC360" class="line">	<span class="n">INITPFN</span><span class="p">(</span><span class="n">clEnqueueAcquireD3D10ObjectsKHR</span><span class="p">);</span></span>
<span id="LC361" class="line">	<span class="n">INITPFN</span><span class="p">(</span><span class="n">clEnqueueReleaseD3D10ObjectsKHR</span><span class="p">);</span></span>
<span id="LC362" class="line"></span>
<span id="LC363" class="line">	<span class="c1">// </span></span>
<span id="LC364" class="line">	<span class="c1">// Given a particular OpenCL platform that has D3D sharing capability, this </span></span>
<span id="LC365" class="line">	<span class="c1">// function will give valid cl_device_ids for D3D sharing.</span></span>
<span id="LC366" class="line">	<span class="c1">// We'll use the returned cl_device_id to use to create a context</span></span>
<span id="LC367" class="line">	<span class="c1">//</span></span>
<span id="LC368" class="line">    <span class="n">errNum</span> <span class="o">=</span> <span class="n">clGetDeviceIDsFromD3D10KHR</span><span class="p">(</span></span>
<span id="LC369" class="line">        <span class="n">cpPlatform</span><span class="p">,</span></span>
<span id="LC370" class="line">        <span class="n">CL_D3D10_DEVICE_KHR</span><span class="p">,</span></span>
<span id="LC371" class="line">        <span class="n">g_pD3DDevice</span><span class="p">,</span></span>
<span id="LC372" class="line">        <span class="n">CL_PREFERRED_DEVICES_FOR_D3D10_KHR</span><span class="p">,</span></span>
<span id="LC373" class="line">        <span class="mi">1</span><span class="p">,</span></span>
<span id="LC374" class="line">        <span class="o">&amp;</span><span class="n">cdDevice</span><span class="p">,</span></span>
<span id="LC375" class="line">        <span class="o">&amp;</span><span class="n">num_devices</span><span class="p">);</span></span>
<span id="LC376" class="line"></span>
<span id="LC377" class="line">	<span class="k">if</span> <span class="p">(</span><span class="n">errNum</span> <span class="o">==</span> <span class="n">CL_INVALID_PLATFORM</span><span class="p">)</span> <span class="p">{</span></span>
<span id="LC378" class="line">		<span class="n">printf</span><span class="p">(</span><span class="s">"Invalid Platform: Specified platform is not valid</span><span class="se">\n</span><span class="s">"</span><span class="p">);</span></span>
<span id="LC379" class="line">	<span class="p">}</span> <span class="k">else</span> <span class="k">if</span><span class="p">(</span> <span class="n">errNum</span> <span class="o">==</span> <span class="n">CL_INVALID_VALUE</span><span class="p">)</span> <span class="p">{</span></span>
<span id="LC380" class="line">		<span class="n">printf</span><span class="p">(</span><span class="s">"Invalid Value: d3d_device_source, d3d_device_set is not valid or num_entries = 0 and devices != NULL or num_devices == devices == NULL</span><span class="se">\n</span><span class="s">"</span><span class="p">);</span></span>
<span id="LC381" class="line">	<span class="p">}</span> <span class="k">else</span> <span class="k">if</span><span class="p">(</span> <span class="n">errNum</span> <span class="o">==</span> <span class="n">CL_DEVICE_NOT_FOUND</span><span class="p">)</span> <span class="p">{</span></span>
<span id="LC382" class="line">		<span class="n">printf</span><span class="p">(</span><span class="s">"No OpenCL devices corresponding to the d3d_object were found</span><span class="se">\n</span><span class="s">"</span><span class="p">);</span></span>
<span id="LC383" class="line">	<span class="p">}</span></span>
<span id="LC384" class="line"></span>
<span id="LC385" class="line">	<span class="c1">//</span></span>
<span id="LC386" class="line">    <span class="c1">// Next, create an OpenCL context on the OpenCL device ID returned </span></span>
<span id="LC387" class="line">	<span class="c1">// above (cl_device_id cdDevice)</span></span>
<span id="LC388" class="line">	<span class="c1">//</span></span>
<span id="LC389" class="line">	<span class="c1">// First set the context to include the D3D device being used</span></span>
<span id="LC390" class="line">    <span class="n">cl_context_properties</span> <span class="n">contextProperties</span><span class="p">[]</span> <span class="o">=</span></span>
<span id="LC391" class="line">    <span class="p">{</span></span>
<span id="LC392" class="line">		<span class="n">CL_CONTEXT_D3D10_DEVICE_KHR</span><span class="p">,</span> <span class="p">(</span><span class="n">cl_context_properties</span><span class="p">)</span><span class="n">g_pD3DDevice</span><span class="p">,</span></span>
<span id="LC393" class="line">        <span class="n">CL_CONTEXT_PLATFORM</span><span class="p">,</span></span>
<span id="LC394" class="line">        <span class="p">(</span><span class="n">cl_context_properties</span><span class="p">)</span><span class="n">cpPlatform</span><span class="p">,</span></span>
<span id="LC395" class="line">        <span class="mi">0</span></span>
<span id="LC396" class="line">    <span class="p">};</span></span>
<span id="LC397" class="line"></span>
<span id="LC398" class="line"></span>
<span id="LC399" class="line">	<span class="c1">//</span></span>
<span id="LC400" class="line">	<span class="c1">// Create the context on the appropriate cl_device_id</span></span>
<span id="LC401" class="line">	<span class="c1">// </span></span>
<span id="LC402" class="line">	<span class="n">context</span> <span class="o">=</span> <span class="n">clCreateContext</span><span class="p">(</span> <span class="n">contextProperties</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">cdDevice</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">errNum</span> <span class="p">)</span> <span class="p">;</span></span>
<span id="LC403" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">errNum</span> <span class="o">!=</span> <span class="n">CL_SUCCESS</span><span class="p">)</span></span>
<span id="LC404" class="line">    <span class="p">{</span></span>
<span id="LC405" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cout</span> <span class="o">&lt;&lt;</span> <span class="s">"Could not create GPU context."</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC406" class="line">    <span class="p">}</span></span>
<span id="LC407" class="line"></span>
<span id="LC408" class="line">	<span class="c1">//</span></span>
<span id="LC409" class="line">	<span class="c1">// Create a command queue on the device</span></span>
<span id="LC410" class="line">	<span class="c1">//</span></span>
<span id="LC411" class="line">    <span class="n">commandQueue</span> <span class="o">=</span> <span class="n">clCreateCommandQueue</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="n">cdDevice</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">);</span></span>
<span id="LC412" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">commandQueue</span> <span class="o">==</span> <span class="nb">NULL</span><span class="p">)</span></span>
<span id="LC413" class="line">    <span class="p">{</span></span>
<span id="LC414" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Failed to create commandQueue for device 0"</span><span class="p">;</span></span>
<span id="LC415" class="line">        <span class="k">return</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC416" class="line">    <span class="p">}</span></span>
<span id="LC417" class="line">	<span class="c1">// </span></span>
<span id="LC418" class="line">	<span class="c1">// Create the texture.  We can now use this context for D3D sharing.</span></span>
<span id="LC419" class="line">	<span class="c1">//</span></span>
<span id="LC420" class="line">	<span class="k">if</span><span class="p">(</span> <span class="n">InitTextures</span><span class="p">(</span><span class="n">context</span><span class="p">)</span> <span class="o">!=</span> <span class="n">S_OK</span> <span class="p">)</span> <span class="p">{</span></span>
<span id="LC421" class="line">		<span class="n">printf</span><span class="p">(</span><span class="s">"Failed to initialize the D3D/OCL texture.</span><span class="se">\n</span><span class="s">"</span><span class="p">);</span></span>
<span id="LC422" class="line">	<span class="p">}</span></span>
<span id="LC423" class="line"></span>
<span id="LC424" class="line">	<span class="c1">//</span></span>
<span id="LC425" class="line">	<span class="c1">// Create OpenCL program from D3Dinterop.cl kernel source</span></span>
<span id="LC426" class="line">	<span class="c1">//</span></span>
<span id="LC427" class="line">    <span class="n">program</span> <span class="o">=</span> <span class="n">CreateProgram</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="n">cdDevice</span><span class="p">,</span> <span class="s">"D3Dinterop.cl"</span><span class="p">);</span></span>
<span id="LC428" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">program</span> <span class="o">==</span> <span class="nb">NULL</span><span class="p">)</span></span>
<span id="LC429" class="line">    <span class="p">{</span></span>
<span id="LC430" class="line">		<span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Failed to open or compile D3Dinterop.cl"</span> <span class="o">&lt;&lt;</span><span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC431" class="line">        <span class="n">Cleanup</span><span class="p">();</span></span>
<span id="LC432" class="line">        <span class="k">return</span> <span class="mi">1</span><span class="p">;</span></span>
<span id="LC433" class="line">    <span class="p">}</span></span>
<span id="LC434" class="line"></span>
<span id="LC435" class="line">	<span class="c1">//</span></span>
<span id="LC436" class="line">	<span class="c1">// Create the texture processing kernel</span></span>
<span id="LC437" class="line">	<span class="c1">//</span></span>
<span id="LC438" class="line">	<span class="n">tex_kernel</span> <span class="o">=</span> <span class="n">clCreateKernel</span><span class="p">(</span><span class="n">program</span><span class="p">,</span> <span class="s">"xyz_init_texture_kernel"</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">);</span></span>
<span id="LC439" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">tex_kernel</span> <span class="o">==</span> <span class="nb">NULL</span><span class="p">)</span></span>
<span id="LC440" class="line">    <span class="p">{</span></span>
<span id="LC441" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Failed to create kernel"</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC442" class="line">        <span class="n">Cleanup</span><span class="p">();</span></span>
<span id="LC443" class="line">        <span class="k">return</span> <span class="mi">1</span><span class="p">;</span></span>
<span id="LC444" class="line">    <span class="p">}</span></span>
<span id="LC445" class="line"></span>
<span id="LC446" class="line">	<span class="c1">//</span></span>
<span id="LC447" class="line">	<span class="c1">// Create the buffer processing kernel</span></span>
<span id="LC448" class="line">	<span class="c1">//</span></span>
<span id="LC449" class="line">	<span class="n">buffer_kernel</span> <span class="o">=</span> <span class="n">clCreateKernel</span><span class="p">(</span><span class="n">program</span><span class="p">,</span> <span class="s">"init_vbo_kernel"</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">);</span></span>
<span id="LC450" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">buffer_kernel</span> <span class="o">==</span> <span class="nb">NULL</span><span class="p">)</span></span>
<span id="LC451" class="line">    <span class="p">{</span></span>
<span id="LC452" class="line">        <span class="n">std</span><span class="o">::</span><span class="n">cerr</span> <span class="o">&lt;&lt;</span> <span class="s">"Failed to create kernel"</span> <span class="o">&lt;&lt;</span> <span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC453" class="line">        <span class="n">Cleanup</span><span class="p">();</span></span>
<span id="LC454" class="line">        <span class="k">return</span> <span class="mi">1</span><span class="p">;</span></span>
<span id="LC455" class="line">    <span class="p">}</span></span>
<span id="LC456" class="line">	<span class="n">computeTexture</span><span class="p">();</span></span>
<span id="LC457" class="line"></span>
<span id="LC458" class="line">	<span class="n">printf</span><span class="p">(</span><span class="s">"Initialized D3D/OpenCL sharing.</span><span class="se">\n</span><span class="s">"</span><span class="p">);</span></span>
<span id="LC459" class="line"></span>
<span id="LC460" class="line">    <span class="c1">// Main message loop</span></span>
<span id="LC461" class="line">    <span class="n">MSG</span> <span class="n">msg</span> <span class="o">=</span> <span class="p">{</span><span class="mi">0</span><span class="p">};</span></span>
<span id="LC462" class="line">    <span class="k">while</span><span class="p">(</span> <span class="n">WM_QUIT</span> <span class="o">!=</span> <span class="n">msg</span><span class="p">.</span><span class="n">message</span> <span class="p">)</span></span>
<span id="LC463" class="line">    <span class="p">{</span></span>
<span id="LC464" class="line">        <span class="k">if</span><span class="p">(</span> <span class="n">PeekMessage</span><span class="p">(</span> <span class="o">&amp;</span><span class="n">msg</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="n">PM_REMOVE</span> <span class="p">)</span> <span class="p">)</span></span>
<span id="LC465" class="line">        <span class="p">{</span></span>
<span id="LC466" class="line">            <span class="n">TranslateMessage</span><span class="p">(</span> <span class="o">&amp;</span><span class="n">msg</span> <span class="p">);</span></span>
<span id="LC467" class="line">            <span class="n">DispatchMessage</span><span class="p">(</span> <span class="o">&amp;</span><span class="n">msg</span> <span class="p">);</span></span>
<span id="LC468" class="line">        <span class="p">}</span></span>
<span id="LC469" class="line">        <span class="k">else</span></span>
<span id="LC470" class="line">        <span class="p">{</span></span>
<span id="LC471" class="line">            <span class="n">Render</span><span class="p">();</span></span>
<span id="LC472" class="line">        <span class="p">}</span></span>
<span id="LC473" class="line">    <span class="p">}</span></span>
<span id="LC474" class="line">	</span>
<span id="LC475" class="line">    <span class="k">return</span> <span class="p">(</span> <span class="kt">int</span> <span class="p">)</span><span class="n">msg</span><span class="p">.</span><span class="n">wParam</span><span class="p">;</span></span>
<span id="LC476" class="line"><span class="p">}</span></span>
<span id="LC477" class="line"></span>
<span id="LC478" class="line"><span class="c1">//-----------------------------------------------------------------------------</span></span>
<span id="LC479" class="line"><span class="c1">// Name: MsgProc()</span></span>
<span id="LC480" class="line"><span class="c1">// Desc: The window's message handler</span></span>
<span id="LC481" class="line"><span class="c1">//-----------------------------------------------------------------------------</span></span>
<span id="LC482" class="line"><span class="kt">bool</span> <span class="n">g_bDone</span> <span class="o">=</span> <span class="nb">false</span><span class="p">;</span></span>
<span id="LC483" class="line"><span class="k">static</span> <span class="n">LRESULT</span> <span class="n">WINAPI</span> <span class="nf">MsgProc</span><span class="p">(</span><span class="n">HWND</span> <span class="n">hWnd</span><span class="p">,</span> <span class="n">UINT</span> <span class="n">msg</span><span class="p">,</span> <span class="n">WPARAM</span> <span class="n">wParam</span><span class="p">,</span> <span class="n">LPARAM</span> <span class="n">lParam</span><span class="p">)</span></span>
<span id="LC484" class="line"><span class="p">{</span></span>
<span id="LC485" class="line">    <span class="k">switch</span><span class="p">(</span><span class="n">msg</span><span class="p">)</span></span>
<span id="LC486" class="line">    <span class="p">{</span></span>
<span id="LC487" class="line">        <span class="k">case</span> <span class="n">WM_KEYDOWN</span><span class="p">:</span></span>
<span id="LC488" class="line">            <span class="k">if</span><span class="p">(</span><span class="n">wParam</span><span class="o">==</span><span class="n">VK_ESCAPE</span><span class="p">)</span> </span>
<span id="LC489" class="line">			<span class="p">{</span></span>
<span id="LC490" class="line">				<span class="n">g_bDone</span> <span class="o">=</span> <span class="nb">true</span><span class="p">;</span></span>
<span id="LC491" class="line">                <span class="n">Cleanup</span><span class="p">();</span></span>
<span id="LC492" class="line">	            <span class="n">PostQuitMessage</span><span class="p">(</span><span class="mi">0</span><span class="p">);</span></span>
<span id="LC493" class="line">				<span class="k">return</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC494" class="line">			<span class="p">}</span></span>
<span id="LC495" class="line">            <span class="k">break</span><span class="p">;</span></span>
<span id="LC496" class="line">        <span class="k">case</span> <span class="n">WM_DESTROY</span><span class="p">:</span></span>
<span id="LC497" class="line">			<span class="n">g_bDone</span> <span class="o">=</span> <span class="nb">true</span><span class="p">;</span></span>
<span id="LC498" class="line">            <span class="n">Cleanup</span><span class="p">();</span></span>
<span id="LC499" class="line">            <span class="n">PostQuitMessage</span><span class="p">(</span><span class="mi">0</span><span class="p">);</span></span>
<span id="LC500" class="line">            <span class="k">return</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC501" class="line">        <span class="k">case</span> <span class="n">WM_PAINT</span><span class="p">:</span></span>
<span id="LC502" class="line">            <span class="n">ValidateRect</span><span class="p">(</span><span class="n">hWnd</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">);</span></span>
<span id="LC503" class="line">            <span class="k">return</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC504" class="line">    <span class="p">}</span></span>
<span id="LC505" class="line">    <span class="k">return</span> <span class="n">DefWindowProc</span><span class="p">(</span><span class="n">hWnd</span><span class="p">,</span> <span class="n">msg</span><span class="p">,</span> <span class="n">wParam</span><span class="p">,</span> <span class="n">lParam</span><span class="p">);</span></span>
<span id="LC506" class="line"><span class="p">}</span></span>
<span id="LC507" class="line"><span class="c1">//--------------------------------------------------------------------------------------</span></span>
<span id="LC508" class="line"><span class="c1">// Register class and create window</span></span>
<span id="LC509" class="line"><span class="c1">//--------------------------------------------------------------------------------------</span></span>
<span id="LC510" class="line"><span class="n">HRESULT</span> <span class="nf">InitWindow</span><span class="p">(</span> <span class="n">HINSTANCE</span> <span class="n">hInstance</span><span class="p">,</span> <span class="kt">int</span> <span class="n">nCmdShow</span> <span class="p">)</span></span>
<span id="LC511" class="line"><span class="p">{</span></span>
<span id="LC512" class="line">    <span class="c1">// Register the window class</span></span>
<span id="LC513" class="line">    <span class="n">WNDCLASSEX</span> <span class="n">wc</span> <span class="o">=</span> <span class="p">{</span> <span class="k">sizeof</span><span class="p">(</span><span class="n">WNDCLASSEX</span><span class="p">),</span> <span class="n">CS_CLASSDC</span><span class="p">,</span> <span class="n">MsgProc</span><span class="p">,</span> <span class="mi">0L</span><span class="p">,</span> <span class="mi">0L</span><span class="p">,</span></span>
<span id="LC514" class="line">                      <span class="n">GetModuleHandle</span><span class="p">(</span><span class="nb">NULL</span><span class="p">),</span> <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span></span>
<span id="LC515" class="line">                      <span class="s">L"OpenCL/D3D10 Texture InterOP"</span><span class="p">,</span> <span class="nb">NULL</span> <span class="p">};</span></span>
<span id="LC516" class="line">    <span class="k">if</span><span class="p">(</span> <span class="o">!</span><span class="n">RegisterClassEx</span><span class="p">(</span> <span class="o">&amp;</span><span class="n">wc</span><span class="p">)</span> <span class="p">)</span></span>
<span id="LC517" class="line">        <span class="k">return</span> <span class="n">E_FAIL</span><span class="p">;</span></span>
<span id="LC518" class="line"></span>
<span id="LC519" class="line">	<span class="kt">int</span> <span class="n">xBorder</span> <span class="o">=</span> <span class="o">::</span><span class="n">GetSystemMetrics</span><span class="p">(</span><span class="n">SM_CXSIZEFRAME</span><span class="p">);</span></span>
<span id="LC520" class="line">	<span class="kt">int</span> <span class="n">yMenu</span> <span class="o">=</span> <span class="o">::</span><span class="n">GetSystemMetrics</span><span class="p">(</span><span class="n">SM_CYMENU</span><span class="p">);</span></span>
<span id="LC521" class="line">	<span class="kt">int</span> <span class="n">yBorder</span> <span class="o">=</span> <span class="o">::</span><span class="n">GetSystemMetrics</span><span class="p">(</span><span class="n">SM_CYSIZEFRAME</span><span class="p">);</span></span>
<span id="LC522" class="line"></span>
<span id="LC523" class="line">    <span class="c1">// Create the application's window (padding by window border for uniform BB sizes across OSs)</span></span>
<span id="LC524" class="line">    <span class="n">g_hWnd</span> <span class="o">=</span> <span class="n">CreateWindow</span><span class="p">(</span> <span class="n">wc</span><span class="p">.</span><span class="n">lpszClassName</span><span class="p">,</span> <span class="s">L"OpenCL/D3D10 Texture InterOP"</span><span class="p">,</span></span>
<span id="LC525" class="line">                              <span class="n">WS_OVERLAPPEDWINDOW</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="n">g_WindowWidth</span> <span class="o">+</span> <span class="mi">2</span><span class="o">*</span><span class="n">xBorder</span><span class="p">,</span> <span class="n">g_WindowHeight</span><span class="o">+</span> <span class="mi">2</span><span class="o">*</span><span class="n">yBorder</span><span class="o">+</span><span class="n">yMenu</span><span class="p">,</span></span>
<span id="LC526" class="line">                              <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="n">wc</span><span class="p">.</span><span class="n">hInstance</span><span class="p">,</span> <span class="nb">NULL</span> <span class="p">);</span></span>
<span id="LC527" class="line">    <span class="k">if</span><span class="p">(</span> <span class="o">!</span><span class="n">g_hWnd</span> <span class="p">)</span></span>
<span id="LC528" class="line">        <span class="k">return</span> <span class="n">E_FAIL</span><span class="p">;</span></span>
<span id="LC529" class="line"></span>
<span id="LC530" class="line">    <span class="n">ShowWindow</span><span class="p">(</span> <span class="n">g_hWnd</span><span class="p">,</span> <span class="n">nCmdShow</span> <span class="p">);</span></span>
<span id="LC531" class="line"></span>
<span id="LC532" class="line">    <span class="k">return</span> <span class="n">S_OK</span><span class="p">;</span></span>
<span id="LC533" class="line"><span class="p">}</span></span>
<span id="LC534" class="line"></span>
<span id="LC535" class="line"><span class="n">HRESULT</span> <span class="nf">InitDeviceAndSwapChain</span><span class="p">(</span><span class="kt">int</span> <span class="n">width</span><span class="p">,</span> <span class="kt">int</span> <span class="n">height</span><span class="p">)</span></span>
<span id="LC536" class="line"><span class="p">{</span></span>
<span id="LC537" class="line">	<span class="n">HRESULT</span> <span class="n">hr</span><span class="p">;</span></span>
<span id="LC538" class="line">	<span class="n">DXGI_SWAP_CHAIN_DESC</span> <span class="n">sd</span><span class="p">;</span></span>
<span id="LC539" class="line">	<span class="n">ZeroMemory</span><span class="p">(</span> <span class="o">&amp;</span><span class="n">sd</span><span class="p">,</span> <span class="k">sizeof</span><span class="p">(</span> <span class="n">sd</span> <span class="p">)</span> <span class="p">);</span></span>
<span id="LC540" class="line">	<span class="n">sd</span><span class="p">.</span><span class="n">BufferCount</span> <span class="o">=</span> <span class="mi">1</span><span class="p">;</span></span>
<span id="LC541" class="line">	<span class="n">sd</span><span class="p">.</span><span class="n">BufferDesc</span><span class="p">.</span><span class="n">Width</span> <span class="o">=</span> <span class="n">width</span><span class="p">;</span></span>
<span id="LC542" class="line">	<span class="n">sd</span><span class="p">.</span><span class="n">BufferDesc</span><span class="p">.</span><span class="n">Height</span> <span class="o">=</span> <span class="n">height</span><span class="p">;</span></span>
<span id="LC543" class="line">	<span class="n">sd</span><span class="p">.</span><span class="n">BufferDesc</span><span class="p">.</span><span class="n">Format</span> <span class="o">=</span> <span class="n">DXGI_FORMAT_R8G8B8A8_UNORM</span><span class="p">;</span></span>
<span id="LC544" class="line">	<span class="n">sd</span><span class="p">.</span><span class="n">BufferDesc</span><span class="p">.</span><span class="n">RefreshRate</span><span class="p">.</span><span class="n">Numerator</span> <span class="o">=</span> <span class="mi">60</span><span class="p">;</span></span>
<span id="LC545" class="line">	<span class="n">sd</span><span class="p">.</span><span class="n">BufferDesc</span><span class="p">.</span><span class="n">RefreshRate</span><span class="p">.</span><span class="n">Denominator</span> <span class="o">=</span> <span class="mi">1</span><span class="p">;</span></span>
<span id="LC546" class="line">	<span class="n">sd</span><span class="p">.</span><span class="n">BufferUsage</span> <span class="o">=</span> <span class="n">DXGI_USAGE_RENDER_TARGET_OUTPUT</span><span class="p">;</span></span>
<span id="LC547" class="line">	<span class="n">sd</span><span class="p">.</span><span class="n">OutputWindow</span> <span class="o">=</span> <span class="n">g_hWnd</span><span class="p">;</span></span>
<span id="LC548" class="line">	<span class="n">sd</span><span class="p">.</span><span class="n">SampleDesc</span><span class="p">.</span><span class="n">Count</span> <span class="o">=</span> <span class="mi">1</span><span class="p">;</span></span>
<span id="LC549" class="line">	<span class="n">sd</span><span class="p">.</span><span class="n">SampleDesc</span><span class="p">.</span><span class="n">Quality</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC550" class="line">	<span class="n">sd</span><span class="p">.</span><span class="n">Windowed</span> <span class="o">=</span> <span class="n">TRUE</span><span class="p">;</span></span>
<span id="LC551" class="line">	<span class="n">D3D10_DRIVER_TYPE</span> <span class="n">g_driverType</span> <span class="o">=</span> <span class="n">D3D10_DRIVER_TYPE_HARDWARE</span><span class="p">;</span></span>
<span id="LC552" class="line">	<span class="n">UINT</span> <span class="n">createDeviceFlags</span> <span class="o">=</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC553" class="line"></span>
<span id="LC554" class="line">	<span class="n">hr</span> <span class="o">=</span> <span class="n">D3D10CreateDeviceAndSwapChain</span><span class="p">(</span> </span>
<span id="LC555" class="line">		<span class="nb">NULL</span><span class="p">,</span> </span>
<span id="LC556" class="line">		<span class="n">g_driverType</span><span class="p">,</span> </span>
<span id="LC557" class="line">		<span class="nb">NULL</span><span class="p">,</span> </span>
<span id="LC558" class="line">		<span class="n">createDeviceFlags</span><span class="p">,</span> </span>
<span id="LC559" class="line">		<span class="n">D3D10_SDK_VERSION</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">sd</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">g_pSwapChain</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">g_pD3DDevice</span><span class="p">);</span></span>
<span id="LC560" class="line"></span>
<span id="LC561" class="line">	<span class="k">if</span><span class="p">(</span> <span class="n">SUCCEEDED</span><span class="p">(</span> <span class="n">hr</span> <span class="p">)</span> <span class="p">)</span> <span class="p">{</span>	</span>
<span id="LC562" class="line">		<span class="n">std</span><span class="o">::</span><span class="n">cout</span><span class="o">&lt;&lt;</span><span class="s">"Created D3D10 Hardware device."</span><span class="o">&lt;&lt;</span><span class="n">std</span><span class="o">::</span><span class="n">endl</span><span class="p">;</span></span>
<span id="LC563" class="line">	<span class="p">}</span></span>
<span id="LC564" class="line">	<span class="k">return</span> <span class="n">hr</span><span class="p">;</span></span>
<span id="LC565" class="line"><span class="p">}</span></span>
<span id="LC566" class="line"><span class="c1">///</span></span>
<span id="LC567" class="line"><span class="c1">// ..creates a render target view of the swap chain back buffer.</span></span>
<span id="LC568" class="line"><span class="c1">// Also it will setup the vertex shader and triangle strip for</span></span>
<span id="LC569" class="line"><span class="c1">// drawing 2 onscreen triangles that form a quad.</span></span>
<span id="LC570" class="line"><span class="n">HRESULT</span> <span class="nf">createRenderTargetViewOfSwapChainBackBuffer</span><span class="p">(</span><span class="kt">int</span> <span class="n">width</span><span class="p">,</span> <span class="kt">int</span> <span class="n">height</span><span class="p">)</span> </span>
<span id="LC571" class="line"><span class="p">{</span></span>
<span id="LC572" class="line">	<span class="n">HRESULT</span> <span class="n">hr</span> <span class="o">=</span> <span class="n">S_OK</span><span class="p">;</span></span>
<span id="LC573" class="line">	<span class="n">ID3D10Texture2D</span><span class="o">*</span> <span class="n">pBackBuffer</span> <span class="o">=</span> <span class="nb">NULL</span><span class="p">;</span></span>
<span id="LC574" class="line">	<span class="k">if</span><span class="p">(</span><span class="n">FAILED</span><span class="p">(</span><span class="n">hr</span> <span class="o">=</span> <span class="n">g_pSwapChain</span><span class="o">-&gt;</span><span class="n">GetBuffer</span><span class="p">(</span> <span class="mi">0</span><span class="p">,</span> <span class="kr">__uuidof</span><span class="p">(</span> <span class="o">*</span><span class="n">pBackBuffer</span> <span class="p">),</span> <span class="p">(</span> <span class="n">LPVOID</span><span class="o">*</span> <span class="p">)</span><span class="o">&amp;</span><span class="n">pBackBuffer</span> <span class="p">)))</span></span>
<span id="LC575" class="line">	<span class="p">{</span></span>
<span id="LC576" class="line">		<span class="n">MessageBox</span><span class="p">(</span><span class="nb">NULL</span><span class="p">,</span><span class="s">L"m_pSwapChain-&gt;GetBuffer failed."</span><span class="p">,</span><span class="s">L"Swap Chain Error"</span><span class="p">,</span> <span class="n">MB_OK</span><span class="p">);</span></span>
<span id="LC577" class="line">		<span class="k">return</span> <span class="n">hr</span><span class="p">;</span></span>
<span id="LC578" class="line">	<span class="p">}</span>	</span>
<span id="LC579" class="line"></span>
<span id="LC580" class="line">	<span class="k">if</span><span class="p">(</span><span class="n">FAILED</span><span class="p">(</span><span class="n">hr</span> <span class="o">=</span> <span class="n">g_pD3DDevice</span><span class="o">-&gt;</span><span class="n">CreateRenderTargetView</span><span class="p">(</span> <span class="n">pBackBuffer</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">g_pRenderTargetView</span> <span class="p">)))</span></span>
<span id="LC581" class="line">	<span class="p">{</span></span>
<span id="LC582" class="line">		<span class="n">MessageBox</span><span class="p">(</span><span class="nb">NULL</span><span class="p">,</span><span class="s">L"m_pDevice-&gt;CreateRenderTargetView failed."</span><span class="p">,</span><span class="s">L"Create Render Tgt View Error"</span><span class="p">,</span> <span class="n">MB_OK</span><span class="p">);</span></span>
<span id="LC583" class="line">		<span class="k">return</span> <span class="n">hr</span><span class="p">;</span></span>
<span id="LC584" class="line">	<span class="p">}</span></span>
<span id="LC585" class="line"></span>
<span id="LC586" class="line">	<span class="n">SAFE_RELEASE</span><span class="p">(</span><span class="n">pBackBuffer</span><span class="p">);</span></span>
<span id="LC587" class="line"></span>
<span id="LC588" class="line">	<span class="n">g_pD3DDevice</span><span class="o">-&gt;</span><span class="n">OMSetRenderTargets</span><span class="p">(</span> <span class="mi">1</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">g_pRenderTargetView</span><span class="p">,</span> <span class="nb">NULL</span> <span class="p">);</span></span>
<span id="LC589" class="line"></span>
<span id="LC590" class="line"></span>
<span id="LC591" class="line">	<span class="c1">// Setup the viewport</span></span>
<span id="LC592" class="line">	<span class="n">D3D10_VIEWPORT</span> <span class="n">vp</span><span class="p">;</span></span>
<span id="LC593" class="line">	<span class="n">vp</span><span class="p">.</span><span class="n">Width</span>		<span class="o">=</span> <span class="p">(</span><span class="n">UINT</span><span class="p">)</span><span class="n">width</span><span class="p">;</span></span>
<span id="LC594" class="line">	<span class="n">vp</span><span class="p">.</span><span class="n">Height</span>		<span class="o">=</span> <span class="p">(</span><span class="n">UINT</span><span class="p">)</span><span class="n">height</span><span class="p">;</span></span>
<span id="LC595" class="line">	<span class="n">vp</span><span class="p">.</span><span class="n">MinDepth</span>		<span class="o">=</span> <span class="mf">0.0</span><span class="n">f</span><span class="p">;</span></span>
<span id="LC596" class="line">	<span class="n">vp</span><span class="p">.</span><span class="n">MaxDepth</span>		<span class="o">=</span> <span class="mf">1.0</span><span class="n">f</span><span class="p">;</span></span>
<span id="LC597" class="line">	<span class="n">vp</span><span class="p">.</span><span class="n">TopLeftX</span>		<span class="o">=</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC598" class="line">	<span class="n">vp</span><span class="p">.</span><span class="n">TopLeftY</span>		<span class="o">=</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC599" class="line">	<span class="n">g_pD3DDevice</span><span class="o">-&gt;</span><span class="n">RSSetViewports</span><span class="p">(</span> <span class="mi">1</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">vp</span> <span class="p">);</span></span>
<span id="LC600" class="line"></span>
<span id="LC601" class="line">    <span class="c1">// Create the effect</span></span>
<span id="LC602" class="line">    <span class="n">DWORD</span> <span class="n">dwShaderFlags</span> <span class="o">=</span> <span class="n">D3D10_SHADER_ENABLE_STRICTNESS</span><span class="p">;</span></span>
<span id="LC603" class="line"><span class="c1">//#if defined( DEBUG ) || defined( _DEBUG )</span></span>
<span id="LC604" class="line">    <span class="c1">// Set the D3D10_SHADER_DEBUG flag to embed debug information in the shaders.</span></span>
<span id="LC605" class="line">    <span class="c1">// Setting this flag improves the shader debugging experience, but still allows </span></span>
<span id="LC606" class="line">    <span class="c1">// the shaders to be optimized and to run exactly the way they will run in </span></span>
<span id="LC607" class="line">    <span class="c1">// the release configuration of this program.</span></span>
<span id="LC608" class="line">    <span class="n">dwShaderFlags</span> <span class="o">|=</span> <span class="n">D3D10_SHADER_DEBUG</span><span class="p">;</span></span>
<span id="LC609" class="line"> <span class="c1">//   #endif</span></span>
<span id="LC610" class="line">    <span class="n">hr</span> <span class="o">=</span> <span class="n">D3DX10CreateEffectFromFile</span><span class="p">(</span> <span class="s">L"D3Dinterop.fx"</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="s">"fx_4_0"</span><span class="p">,</span> <span class="n">dwShaderFlags</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span></span>
<span id="LC611" class="line">                                         <span class="n">g_pD3DDevice</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">g_pEffect</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="nb">NULL</span> <span class="p">);</span></span>
<span id="LC612" class="line">    <span class="k">if</span><span class="p">(</span> <span class="n">FAILED</span><span class="p">(</span> <span class="n">hr</span> <span class="p">)</span> <span class="p">)</span></span>
<span id="LC613" class="line">    <span class="p">{</span></span>
<span id="LC614" class="line">        <span class="n">MessageBox</span><span class="p">(</span> <span class="nb">NULL</span><span class="p">,</span></span>
<span id="LC615" class="line">                    <span class="s">L"The FX file (D3Dinterop.fx) cannot be located.  Please run this executable from the directory that contains the FX file."</span><span class="p">,</span> <span class="s">L"Error"</span><span class="p">,</span> <span class="n">MB_OK</span> <span class="p">);</span></span>
<span id="LC616" class="line">        <span class="k">return</span> <span class="n">hr</span><span class="p">;</span></span>
<span id="LC617" class="line">    <span class="p">}</span></span>
<span id="LC618" class="line"></span>
<span id="LC619" class="line">    <span class="c1">// Obtain the technique</span></span>
<span id="LC620" class="line">    <span class="n">g_pTechnique</span> <span class="o">=</span> <span class="n">g_pEffect</span><span class="o">-&gt;</span><span class="n">GetTechniqueByName</span><span class="p">(</span> <span class="s">"Render"</span> <span class="p">);</span>	</span>
<span id="LC621" class="line"></span>
<span id="LC622" class="line">    <span class="c1">// Define the input layout</span></span>
<span id="LC623" class="line">    <span class="n">D3D10_INPUT_ELEMENT_DESC</span> <span class="n">layout</span><span class="p">[]</span> <span class="o">=</span></span>
<span id="LC624" class="line">    <span class="p">{</span></span>
<span id="LC625" class="line">        <span class="p">{</span> <span class="s">"POSITION"</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="n">DXGI_FORMAT_R32G32B32_FLOAT</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="n">D3D10_INPUT_PER_VERTEX_DATA</span><span class="p">,</span> <span class="mi">0</span> <span class="p">},</span></span>
<span id="LC626" class="line">	    <span class="p">{</span> <span class="s">"TEXCOORD"</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="n">DXGI_FORMAT_R32G32_FLOAT</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">12</span><span class="p">,</span> <span class="n">D3D10_INPUT_PER_VERTEX_DATA</span><span class="p">,</span> <span class="mi">0</span> <span class="p">},</span> </span>
<span id="LC627" class="line">    <span class="p">};</span></span>
<span id="LC628" class="line">    <span class="n">UINT</span> <span class="n">numElements</span> <span class="o">=</span> <span class="k">sizeof</span><span class="p">(</span> <span class="n">layout</span> <span class="p">)</span> <span class="o">/</span> <span class="k">sizeof</span><span class="p">(</span> <span class="n">layout</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="p">);</span></span>
<span id="LC629" class="line"></span>
<span id="LC630" class="line">    <span class="c1">// Create the input layout</span></span>
<span id="LC631" class="line">    <span class="n">D3D10_PASS_DESC</span> <span class="n">PassDesc</span><span class="p">;</span></span>
<span id="LC632" class="line">    <span class="n">g_pTechnique</span><span class="o">-&gt;</span><span class="n">GetPassByIndex</span><span class="p">(</span> <span class="mi">0</span> <span class="p">)</span><span class="o">-&gt;</span><span class="n">GetDesc</span><span class="p">(</span> <span class="o">&amp;</span><span class="n">PassDesc</span> <span class="p">);</span></span>
<span id="LC633" class="line">    <span class="n">hr</span> <span class="o">=</span> <span class="n">g_pD3DDevice</span><span class="o">-&gt;</span><span class="n">CreateInputLayout</span><span class="p">(</span> <span class="n">layout</span><span class="p">,</span> <span class="n">numElements</span><span class="p">,</span> <span class="n">PassDesc</span><span class="p">.</span><span class="n">pIAInputSignature</span><span class="p">,</span></span>
<span id="LC634" class="line">                                          <span class="n">PassDesc</span><span class="p">.</span><span class="n">IAInputSignatureSize</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">g_pVertexLayout</span> <span class="p">);</span></span>
<span id="LC635" class="line">    <span class="k">if</span><span class="p">(</span> <span class="n">FAILED</span><span class="p">(</span> <span class="n">hr</span> <span class="p">)</span> <span class="p">)</span></span>
<span id="LC636" class="line">        <span class="k">return</span> <span class="n">hr</span><span class="p">;</span></span>
<span id="LC637" class="line"></span>
<span id="LC638" class="line">    <span class="c1">// Set the input layout</span></span>
<span id="LC639" class="line">    <span class="n">g_pD3DDevice</span><span class="o">-&gt;</span><span class="n">IASetInputLayout</span><span class="p">(</span> <span class="n">g_pVertexLayout</span> <span class="p">);</span></span>
<span id="LC640" class="line"></span>
<span id="LC641" class="line">    <span class="c1">// Create vertex buffer</span></span>
<span id="LC642" class="line">    <span class="n">SimpleVertex</span> <span class="n">vertices</span><span class="p">[]</span> <span class="o">=</span></span>
<span id="LC643" class="line">    <span class="p">{</span></span>
<span id="LC644" class="line">		<span class="p">{</span> <span class="n">D3DXVECTOR3</span><span class="p">(</span> <span class="o">-</span><span class="mf">0.5</span><span class="n">f</span><span class="p">,</span> <span class="o">-</span><span class="mf">0.5</span><span class="n">f</span><span class="p">,</span> <span class="mf">0.5</span><span class="n">f</span> <span class="p">),</span> <span class="n">D3DXVECTOR2</span><span class="p">(</span> <span class="mf">0.0</span><span class="n">f</span><span class="p">,</span> <span class="mf">0.0</span><span class="n">f</span> <span class="p">)</span> <span class="p">},</span></span>
<span id="LC645" class="line">		<span class="p">{</span> <span class="n">D3DXVECTOR3</span><span class="p">(</span> <span class="o">-</span><span class="mf">0.5</span><span class="n">f</span><span class="p">,</span> <span class="mf">0.5</span><span class="n">f</span><span class="p">,</span> <span class="mf">0.5</span><span class="n">f</span> <span class="p">),</span> <span class="n">D3DXVECTOR2</span><span class="p">(</span>  <span class="mf">0.0</span><span class="n">f</span><span class="p">,</span> <span class="mf">1.0</span><span class="n">f</span> <span class="p">)</span> <span class="p">},</span></span>
<span id="LC646" class="line">		<span class="p">{</span> <span class="n">D3DXVECTOR3</span><span class="p">(</span> <span class="mf">0.5</span><span class="n">f</span><span class="p">,</span> <span class="o">-</span><span class="mf">0.5</span><span class="n">f</span><span class="p">,</span> <span class="mf">0.5</span><span class="n">f</span> <span class="p">),</span> <span class="n">D3DXVECTOR2</span><span class="p">(</span> <span class="mf">1.0</span><span class="n">f</span><span class="p">,</span> <span class="mf">0.0</span><span class="n">f</span> <span class="p">)</span> <span class="p">},</span></span>
<span id="LC647" class="line">		<span class="p">{</span> <span class="n">D3DXVECTOR3</span><span class="p">(</span>  <span class="mf">0.5</span><span class="n">f</span><span class="p">,</span>  <span class="mf">0.5</span><span class="n">f</span><span class="p">,</span> <span class="mf">0.5</span><span class="n">f</span> <span class="p">),</span> <span class="n">D3DXVECTOR2</span><span class="p">(</span> <span class="mf">1.0</span><span class="n">f</span><span class="p">,</span> <span class="mf">1.0</span><span class="n">f</span> <span class="p">)</span> <span class="p">},</span></span>
<span id="LC648" class="line">	<span class="p">};</span></span>
<span id="LC649" class="line"></span>
<span id="LC650" class="line">    <span class="n">D3D10_BUFFER_DESC</span> <span class="n">bd</span><span class="p">;</span></span>
<span id="LC651" class="line">    <span class="n">bd</span><span class="p">.</span><span class="n">Usage</span> <span class="o">=</span> <span class="n">D3D10_USAGE_DEFAULT</span><span class="p">;</span></span>
<span id="LC652" class="line">    <span class="n">bd</span><span class="p">.</span><span class="n">ByteWidth</span> <span class="o">=</span> <span class="k">sizeof</span><span class="p">(</span> <span class="n">SimpleVertex</span> <span class="p">)</span> <span class="o">*</span> <span class="mi">4</span><span class="p">;</span></span>
<span id="LC653" class="line">    <span class="n">bd</span><span class="p">.</span><span class="n">BindFlags</span> <span class="o">=</span> <span class="n">D3D10_BIND_VERTEX_BUFFER</span><span class="p">;</span></span>
<span id="LC654" class="line">    <span class="n">bd</span><span class="p">.</span><span class="n">CPUAccessFlags</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC655" class="line">    <span class="n">bd</span><span class="p">.</span><span class="n">MiscFlags</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC656" class="line">    <span class="n">D3D10_SUBRESOURCE_DATA</span> <span class="n">InitData</span><span class="p">;</span></span>
<span id="LC657" class="line">    <span class="n">InitData</span><span class="p">.</span><span class="n">pSysMem</span> <span class="o">=</span> <span class="n">vertices</span><span class="p">;</span></span>
<span id="LC658" class="line">    <span class="n">hr</span> <span class="o">=</span> <span class="n">g_pD3DDevice</span><span class="o">-&gt;</span><span class="n">CreateBuffer</span><span class="p">(</span> <span class="o">&amp;</span><span class="n">bd</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">InitData</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">g_pVertexBuffer</span> <span class="p">);</span></span>
<span id="LC659" class="line">    <span class="k">if</span><span class="p">(</span> <span class="n">FAILED</span><span class="p">(</span> <span class="n">hr</span> <span class="p">)</span> <span class="p">)</span></span>
<span id="LC660" class="line">        <span class="k">return</span> <span class="n">hr</span><span class="p">;</span></span>
<span id="LC661" class="line"></span>
<span id="LC662" class="line">    <span class="c1">// Set vertex buffer</span></span>
<span id="LC663" class="line">    <span class="n">UINT</span> <span class="n">stride</span> <span class="o">=</span> <span class="k">sizeof</span><span class="p">(</span> <span class="n">SimpleVertex</span> <span class="p">);</span></span>
<span id="LC664" class="line">    <span class="n">UINT</span> <span class="n">offset</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC665" class="line">    <span class="n">g_pD3DDevice</span><span class="o">-&gt;</span><span class="n">IASetVertexBuffers</span><span class="p">(</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">g_pVertexBuffer</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">stride</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">offset</span> <span class="p">);</span></span>
<span id="LC666" class="line"></span>
<span id="LC667" class="line">    <span class="c1">// Set primitive topology</span></span>
<span id="LC668" class="line">    <span class="c1">//g_pD3DDevice-&gt;IASetPrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST );</span></span>
<span id="LC669" class="line">    <span class="n">g_pD3DDevice</span><span class="o">-&gt;</span><span class="n">IASetPrimitiveTopology</span><span class="p">(</span> <span class="n">D3D10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP</span> <span class="p">);</span></span>
<span id="LC670" class="line">	<span class="n">g_pDiffuseVariable</span> <span class="o">=</span> </span>
<span id="LC671" class="line">		<span class="n">g_pEffect</span><span class="o">-&gt;</span><span class="n">GetVariableByName</span><span class="p">(</span><span class="s">"txDiffuse"</span><span class="p">)</span><span class="o">-&gt;</span><span class="n">AsShaderResource</span><span class="p">();</span></span>
<span id="LC672" class="line"></span>
<span id="LC673" class="line"><span class="c1">/////////////////////////</span></span>
<span id="LC674" class="line">    <span class="c1">// Define the input layout</span></span>
<span id="LC675" class="line">    <span class="n">D3D10_INPUT_ELEMENT_DESC</span> <span class="n">sine_layout</span><span class="p">[]</span> <span class="o">=</span></span>
<span id="LC676" class="line">    <span class="p">{</span></span>
<span id="LC677" class="line">        <span class="p">{</span> <span class="s">"POSITION"</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="n">DXGI_FORMAT_R32G32B32_FLOAT</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="n">D3D10_INPUT_PER_VERTEX_DATA</span><span class="p">,</span> <span class="mi">0</span> <span class="p">},</span></span>
<span id="LC678" class="line">    <span class="p">};</span></span>
<span id="LC679" class="line">    <span class="n">numElements</span> <span class="o">=</span> <span class="k">sizeof</span><span class="p">(</span> <span class="n">sine_layout</span> <span class="p">)</span> <span class="o">/</span> <span class="k">sizeof</span><span class="p">(</span> <span class="n">sine_layout</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="p">);</span></span>
<span id="LC680" class="line"></span>
<span id="LC681" class="line">    <span class="c1">// Create the input layout</span></span>
<span id="LC682" class="line">    <span class="n">g_pTechnique</span><span class="o">-&gt;</span><span class="n">GetPassByIndex</span><span class="p">(</span> <span class="mi">1</span> <span class="p">)</span><span class="o">-&gt;</span><span class="n">GetDesc</span><span class="p">(</span> <span class="o">&amp;</span><span class="n">PassDesc</span> <span class="p">);</span></span>
<span id="LC683" class="line">    <span class="n">hr</span> <span class="o">=</span> <span class="n">g_pD3DDevice</span><span class="o">-&gt;</span><span class="n">CreateInputLayout</span><span class="p">(</span> <span class="n">sine_layout</span><span class="p">,</span> <span class="n">numElements</span><span class="p">,</span> <span class="n">PassDesc</span><span class="p">.</span><span class="n">pIAInputSignature</span><span class="p">,</span></span>
<span id="LC684" class="line">                                          <span class="n">PassDesc</span><span class="p">.</span><span class="n">IAInputSignatureSize</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">g_pSineVertexLayout</span> <span class="p">);</span></span>
<span id="LC685" class="line">    <span class="k">if</span><span class="p">(</span> <span class="n">FAILED</span><span class="p">(</span> <span class="n">hr</span> <span class="p">)</span> <span class="p">)</span></span>
<span id="LC686" class="line">        <span class="k">return</span> <span class="n">hr</span><span class="p">;</span></span>
<span id="LC687" class="line"></span>
<span id="LC688" class="line">    <span class="c1">// Set the input layout</span></span>
<span id="LC689" class="line">   <span class="c1">// g_pD3DDevice-&gt;IASetInputLayout( g_pSineVertexLayout );</span></span>
<span id="LC690" class="line"></span>
<span id="LC691" class="line">    <span class="c1">// Create vertex buffer</span></span>
<span id="LC692" class="line"> <span class="c1">//   SimpleSineVertex sinevertices[256] =</span></span>
<span id="LC693" class="line"> <span class="c1">//   {</span></span>
<span id="LC694" class="line"><span class="c1">//		{ D3DXVECTOR4( -0.75f, -0.75f, 0.5f, 0.0f ) },</span></span>
<span id="LC695" class="line"><span class="c1">//		{ D3DXVECTOR4(  0.75f,  0.75f, 0.5f, 0.0f ) },</span></span>
<span id="LC696" class="line"><span class="c1">//};</span></span>
<span id="LC697" class="line"></span>
<span id="LC698" class="line">    <span class="n">bd</span><span class="p">.</span><span class="n">Usage</span> <span class="o">=</span> <span class="n">D3D10_USAGE_DEFAULT</span><span class="p">;</span></span>
<span id="LC699" class="line">    <span class="n">bd</span><span class="p">.</span><span class="n">ByteWidth</span> <span class="o">=</span> <span class="k">sizeof</span><span class="p">(</span> <span class="n">SimpleSineVertex</span> <span class="p">)</span> <span class="o">*</span> <span class="mi">256</span><span class="p">;</span></span>
<span id="LC700" class="line">    <span class="n">bd</span><span class="p">.</span><span class="n">BindFlags</span> <span class="o">=</span> <span class="n">D3D10_BIND_VERTEX_BUFFER</span><span class="p">;</span></span>
<span id="LC701" class="line">    <span class="n">bd</span><span class="p">.</span><span class="n">CPUAccessFlags</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC702" class="line">    <span class="n">bd</span><span class="p">.</span><span class="n">MiscFlags</span> <span class="o">=</span> <span class="mi">0</span><span class="p">;</span></span>
<span id="LC703" class="line"> <span class="c1">//   InitData.pSysMem = sinevertices;</span></span>
<span id="LC704" class="line">    <span class="n">hr</span> <span class="o">=</span> <span class="n">g_pD3DDevice</span><span class="o">-&gt;</span><span class="n">CreateBuffer</span><span class="p">(</span> <span class="o">&amp;</span><span class="n">bd</span><span class="p">,</span> <span class="nb">NULL</span><span class="p">,</span> <span class="o">&amp;</span><span class="n">g_pSineVertexBuffer</span> <span class="p">);</span></span>
<span id="LC705" class="line">    <span class="k">if</span><span class="p">(</span> <span class="n">FAILED</span><span class="p">(</span> <span class="n">hr</span> <span class="p">)</span> <span class="p">)</span></span>
<span id="LC706" class="line">        <span class="k">return</span> <span class="n">hr</span><span class="p">;</span></span>
<span id="LC707" class="line"><span class="cm">/*</span></span>
<span id="LC708" class="line"><span class="cm">    // Set vertex buffer</span></span>
<span id="LC709" class="line"><span class="cm">    stride = sizeof( SimpleSineVertex );</span></span>
<span id="LC710" class="line"><span class="cm">    offset = 0;</span></span>
<span id="LC711" class="line"><span class="cm">    g_pD3DDevice-&gt;IASetVertexBuffers( 0, 1, &amp;g_pSineVertexBuffer, &amp;stride, &amp;offset );</span></span>
<span id="LC712" class="line"><span class="cm">*/</span></span>
<span id="LC713" class="line">    <span class="k">return</span> <span class="n">S_OK</span><span class="p">;</span>	<span class="c1">//return hr;</span></span>
<span id="LC714" class="line"><span class="p">}</span></span>
<span id="LC715" class="line"></span>
<span id="LC716" class="line"><span class="c1">///</span></span>
<span id="LC717" class="line"><span class="c1">//  Cleanup any created OpenCL resources</span></span>
<span id="LC718" class="line"><span class="c1">//</span></span>
<span id="LC719" class="line"><span class="kt">void</span> <span class="nf">Cleanup</span><span class="p">()</span></span>
<span id="LC720" class="line"><span class="p">{</span></span>
<span id="LC721" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">commandQueue</span> <span class="o">!=</span> <span class="mi">0</span><span class="p">)</span> <span class="n">clReleaseCommandQueue</span><span class="p">(</span><span class="n">commandQueue</span><span class="p">);</span></span>
<span id="LC722" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">tex_kernel</span> <span class="o">!=</span> <span class="mi">0</span><span class="p">)</span> <span class="n">clReleaseKernel</span><span class="p">(</span><span class="n">tex_kernel</span><span class="p">);</span></span>
<span id="LC723" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">program</span> <span class="o">!=</span> <span class="mi">0</span><span class="p">)</span> <span class="n">clReleaseProgram</span><span class="p">(</span><span class="n">program</span><span class="p">);</span></span>
<span id="LC724" class="line">    <span class="k">if</span> <span class="p">(</span><span class="n">context</span> <span class="o">!=</span> <span class="mi">0</span><span class="p">)</span> <span class="n">clReleaseContext</span><span class="p">(</span><span class="n">context</span><span class="p">);</span></span>
<span id="LC725" class="line">	<span class="k">if</span> <span class="p">(</span><span class="n">g_clTexture2D</span> <span class="o">!=</span> <span class="mi">0</span> <span class="p">)</span> <span class="n">clReleaseMemObject</span><span class="p">(</span><span class="n">g_clTexture2D</span><span class="p">);</span></span>
<span id="LC726" class="line"></span>
<span id="LC727" class="line">	<span class="k">if</span><span class="p">(</span> <span class="n">pSRView</span> <span class="o">!=</span> <span class="nb">NULL</span> <span class="p">)</span> <span class="n">pSRView</span><span class="o">-&gt;</span><span class="n">Release</span><span class="p">();</span> </span>
<span id="LC728" class="line">	<span class="k">if</span><span class="p">(</span> <span class="n">g_pTexture2D</span> <span class="o">!=</span> <span class="nb">NULL</span> <span class="p">)</span> <span class="n">g_pTexture2D</span><span class="o">-&gt;</span><span class="n">Release</span><span class="p">();</span></span>
<span id="LC729" class="line">	<span class="k">if</span><span class="p">(</span> <span class="n">g_pVertexLayout</span> <span class="o">!=</span> <span class="nb">NULL</span> <span class="p">)</span> <span class="n">g_pVertexLayout</span><span class="o">-&gt;</span><span class="n">Release</span><span class="p">();</span></span>
<span id="LC730" class="line">	<span class="k">if</span><span class="p">(</span> <span class="n">g_pEffect</span> <span class="o">!=</span> <span class="nb">NULL</span> <span class="p">)</span> <span class="n">g_pEffect</span><span class="o">-&gt;</span><span class="n">Release</span><span class="p">();</span></span>
<span id="LC731" class="line">	<span class="k">if</span><span class="p">(</span> <span class="n">g_pSwapChain</span> <span class="o">!=</span> <span class="nb">NULL</span> <span class="p">)</span> <span class="n">g_pSwapChain</span><span class="o">-&gt;</span><span class="n">Release</span><span class="p">();</span></span>
<span id="LC732" class="line">	<span class="k">if</span><span class="p">(</span> <span class="n">g_pVertexBuffer</span> <span class="o">!=</span> <span class="nb">NULL</span> <span class="p">)</span> <span class="n">g_pVertexBuffer</span><span class="o">-&gt;</span><span class="n">Release</span><span class="p">();</span></span>
<span id="LC733" class="line">	<span class="k">if</span><span class="p">(</span> <span class="n">g_pD3DDevice</span> <span class="o">!=</span> <span class="nb">NULL</span> <span class="p">)</span> <span class="n">g_pD3DDevice</span><span class="o">-&gt;</span><span class="n">Release</span><span class="p">();</span></span>
<span id="LC734" class="line"></span>
<span id="LC735" class="line"></span>
<span id="LC736" class="line">	<span class="n">exit</span><span class="p">(</span><span class="mi">0</span><span class="p">);</span></span>
<span id="LC737" class="line"><span class="p">}</span></span></code></pre>
</div>
</div>


</article>
</div>

</div>
<div class="modal" id="modal-remove-blob">
<div class="modal-dialog">
<div class="modal-content">
<div class="modal-header">
<a class="close" data-dismiss="modal" href="#">×</a>
<h3 class="page-title">Delete D3Dinterop.cpp</h3>
</div>
<div class="modal-body">
<form class="form-horizontal js-replace-blob-form js-quick-submit js-requires-input" action="/EN-605.417.31_FA16/intro_to_gpu/blob/master/opencl-book-samples/src/Chapter_11/D3Dinterop/D3Dinterop.cpp" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="_method" value="delete" /><input type="hidden" name="authenticity_token" value="z7Hugbhh419mHihH/TFT8o07W+IhCv2Ptx6WCuuPoGeDgE0L82RyyMIbxKg6YT7BluKMMNvB3q1IdTeL0kbJcA==" /><div class="form-group commit_message-group">
<label class="control-label" for="commit_message-06d11fe88286586d5b48b6259155b7f0">Commit message
</label><div class="col-sm-10">
<div class="commit-message-container">
<div class="max-width-marker"></div>
<textarea name="commit_message" id="commit_message-06d11fe88286586d5b48b6259155b7f0" class="form-control js-commit-message" placeholder="Delete D3Dinterop.cpp" required="required" rows="3">
Delete D3Dinterop.cpp</textarea>
</div>
</div>
</div>

<div class="form-group branch">
<label class="control-label" for="target_branch">Target branch</label>
<div class="col-sm-10">
<input type="text" name="target_branch" id="target_branch" value="master" required="required" class="form-control js-target-branch" />
<div class="js-create-merge-request-container">
<div class="checkbox">
<label for="create_merge_request-f934bddcd301c321c8d2138f47cdccd2"><input type="checkbox" name="create_merge_request" id="create_merge_request-f934bddcd301c321c8d2138f47cdccd2" value="1" class="js-create-merge-request" checked="checked" />
Start a <strong>new merge request</strong> with these changes
</label></div>
</div>
</div>
</div>
<input type="hidden" name="original_branch" id="original_branch" value="master" class="js-original-branch" />

<div class="form-group">
<div class="col-sm-offset-2 col-sm-10">
<button name="button" type="submit" class="btn btn-remove btn-remove-file">Delete file</button>
<a class="btn btn-cancel" data-dismiss="modal" href="#">Cancel</a>
</div>
</div>
</form></div>
</div>
</div>
</div>
<script>
  new NewCommitForm($('.js-replace-blob-form'))
</script>

<div class="modal" id="modal-upload-blob">
<div class="modal-dialog">
<div class="modal-content">
<div class="modal-header">
<a class="close" data-dismiss="modal" href="#">×</a>
<h3 class="page-title">Replace D3Dinterop.cpp</h3>
</div>
<div class="modal-body">
<form class="js-quick-submit js-upload-blob-form form-horizontal" action="/EN-605.417.31_FA16/intro_to_gpu/update/master/opencl-book-samples/src/Chapter_11/D3Dinterop/D3Dinterop.cpp" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="_method" value="put" /><input type="hidden" name="authenticity_token" value="g5ZtNuEaz3/1FiaSfPH6nGOM0GaqJXu6HZwyfaGMBavPp868qh9e6FETyn27oZeveFUHtFDuWJji95P8mEVsvA==" /><div class="dropzone">
<div class="dropzone-previews blob-upload-dropzone-previews">
<p class="dz-message light">
Attach a file by drag &amp; drop or
<a class="markdown-selector" href="#">click to upload</a>
</p>
</div>
</div>
<br>
<div class="alert alert-danger data dropzone-alerts" style="display:none"></div>
<div class="form-group commit_message-group">
<label class="control-label" for="commit_message-2c94a5dc3f7e7c6defc6ad9190eccecd">Commit message
</label><div class="col-sm-10">
<div class="commit-message-container">
<div class="max-width-marker"></div>
<textarea name="commit_message" id="commit_message-2c94a5dc3f7e7c6defc6ad9190eccecd" class="form-control js-commit-message" placeholder="Replace D3Dinterop.cpp" required="required" rows="3">
Replace D3Dinterop.cpp</textarea>
</div>
</div>
</div>

<div class="form-group branch">
<label class="control-label" for="target_branch">Target branch</label>
<div class="col-sm-10">
<input type="text" name="target_branch" id="target_branch" value="master" required="required" class="form-control js-target-branch" />
<div class="js-create-merge-request-container">
<div class="checkbox">
<label for="create_merge_request-01c5c4866fcabff2a368653bd50913c5"><input type="checkbox" name="create_merge_request" id="create_merge_request-01c5c4866fcabff2a368653bd50913c5" value="1" class="js-create-merge-request" checked="checked" />
Start a <strong>new merge request</strong> with these changes
</label></div>
</div>
</div>
</div>
<input type="hidden" name="original_branch" id="original_branch" value="master" class="js-original-branch" />

<div class="form-actions">
<button name="button" type="submit" class="btn btn-small btn-create btn-upload-file" id="submit-all">Replace file</button>
<a class="btn btn-cancel" data-dismiss="modal" href="#">Cancel</a>
</div>
</form></div>
</div>
</div>
</div>
<script>
  gl.utils.disableButtonIfEmptyField($('.js-upload-blob-form').find('.js-commit-message'), '.btn-upload-file');
  new BlobFileDropzone($('.js-upload-blob-form'), 'put');
  new NewCommitForm($('.js-upload-blob-form'))
</script>

</div>

</div>
</div>
</div>
</div>



</body>
</html>

