<!DOCTYPE html>
<html class="" lang="en">
<head prefix="og: http://ogp.me/ns#">
<meta charset="utf-8">
<meta content="IE=edge" http-equiv="X-UA-Compatible">
<meta content="object" property="og:type">
<meta content="GitLab" property="og:site_name">
<meta content="opencl-book-samples/src/Chapter_8/ImageFilter2D/ImageFilter2D.cl · master · EN-605.417.31_FA16 / intro_to_gpu" property="og:title">
<meta content="Base project for all code to be presented and worked on during the course" property="og:description">
<meta content="http://absaroka.jhuep.com/assets/gitlab_logo-7ae504fe4f68fdebb3c2034e36621930cd36ea87924c11ff65dbcb8ed50dca58.png" property="og:image">
<meta content="http://absaroka.jhuep.com/EN-605.417.31_FA16/intro_to_gpu/blob/master/opencl-book-samples/src/Chapter_8/ImageFilter2D/ImageFilter2D.cl" property="og:url">
<meta content="summary" property="twitter:card">
<meta content="opencl-book-samples/src/Chapter_8/ImageFilter2D/ImageFilter2D.cl · master · EN-605.417.31_FA16 / intro_to_gpu" property="twitter:title">
<meta content="Base project for all code to be presented and worked on during the course" property="twitter:description">
<meta content="http://absaroka.jhuep.com/assets/gitlab_logo-7ae504fe4f68fdebb3c2034e36621930cd36ea87924c11ff65dbcb8ed50dca58.png" property="twitter:image">

<title>opencl-book-samples/src/Chapter_8/ImageFilter2D/ImageFilter2D.cl · master · EN-605.417.31_FA16 / intro_to_gpu · GitLab</title>
<meta content="Base project for all code to be presented and worked on during the course" name="description">
<link rel="shortcut icon" type="image/x-icon" href="/assets/favicon-075eba76312e8421991a0c1f89a89ee81678bcde72319dd3e8047e2a47cd3a42.ico" />
<link rel="stylesheet" media="all" href="/assets/application-b82c159e67a3d15c3f67bf6b7968181447bd0473e3acdf3b874759239ab1296b.css" />
<link rel="stylesheet" media="print" href="/assets/print-9c3a1eb4a2f45c9f3d7dd4de03f14c2e6b921e757168b595d7f161bbc320fc05.css" />
<script src="/assets/application-b6e6a0ec5d9fa435390d9f3cd075c95e666cffbe02f641b8b7cdcd9f3c168ed3.js"></script>
<meta name="csrf-param" content="authenticity_token" />
<meta name="csrf-token" content="q0spfxjIXhzf3x8jHxWcZW8nw1hJHKYiOYv54ZZ6mpzneor1U83Pi3va88zYRfFWdP4UirPXhQDG4Fhgr7Pziw==" />
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
<input type="hidden" name="path" id="path" value="opencl-book-samples/src/Chapter_8/ImageFilter2D/ImageFilter2D.cl" />
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
<a href="/EN-605.417.31_FA16/intro_to_gpu/tree/master/opencl-book-samples/src/Chapter_8">Chapter_8</a>
</li>
<li>
<a href="/EN-605.417.31_FA16/intro_to_gpu/tree/master/opencl-book-samples/src/Chapter_8/ImageFilter2D">ImageFilter2D</a>
</li>
<li>
<a href="/EN-605.417.31_FA16/intro_to_gpu/blob/master/opencl-book-samples/src/Chapter_8/ImageFilter2D/ImageFilter2D.cl"><strong>
ImageFilter2D.cl
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
ImageFilter2D.cl
</strong>
<small>
1.29 KB
</small>
<div class="file-actions hidden-xs">
<div class="btn-group tree-btn-group">
<a class="btn btn-sm" target="_blank" href="/EN-605.417.31_FA16/intro_to_gpu/raw/master/opencl-book-samples/src/Chapter_8/ImageFilter2D/ImageFilter2D.cl">Raw</a>
<a class="btn btn-sm" href="/EN-605.417.31_FA16/intro_to_gpu/blame/master/opencl-book-samples/src/Chapter_8/ImageFilter2D/ImageFilter2D.cl">Blame</a>
<a class="btn btn-sm" href="/EN-605.417.31_FA16/intro_to_gpu/commits/master/opencl-book-samples/src/Chapter_8/ImageFilter2D/ImageFilter2D.cl">History</a>
<a class="btn btn-sm" href="/EN-605.417.31_FA16/intro_to_gpu/blob/18ea4946eebf9c9b710405b0848cb85613d7ac47/opencl-book-samples/src/Chapter_8/ImageFilter2D/ImageFilter2D.cl">Permalink</a>
</div>
<div class="btn-group" role="group">
<a class="btn btn-sm" href="/EN-605.417.31_FA16/intro_to_gpu/edit/master/opencl-book-samples/src/Chapter_8/ImageFilter2D/ImageFilter2D.cl">Edit</a>
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
</div>
<div class="blob-content" data-blob-id="284dfc4f1850ce42af2f482796d8d82e87aa7484">
<pre class="code highlight"><code><span id="LC1" class="line"></span>
<span id="LC2" class="line"><span class="nv">//</span> <span class="nv">Gaussian</span> <span class="nv">filter</span> <span class="nv">of</span> <span class="nv">image</span></span>
<span id="LC3" class="line"></span>
<span id="LC4" class="line"><span class="nv">__kernel</span> <span class="nv">void</span> <span class="nv">gaussian_filter</span><span class="p">(</span><span class="nv">__read_only</span> <span class="nv">image2d_t</span> <span class="nv">srcImg,</span></span>
<span id="LC5" class="line">                              <span class="nv">__write_only</span> <span class="nv">image2d_t</span> <span class="nv">dstImg,</span></span>
<span id="LC6" class="line">                              <span class="nv">sampler_t</span> <span class="nv">sampler,</span></span>
<span id="LC7" class="line">                              <span class="nv">int</span> <span class="nv">width,</span> <span class="nv">int</span> <span class="nv">height</span><span class="p">)</span></span>
<span id="LC8" class="line"><span class="nv">{</span></span>
<span id="LC9" class="line">    <span class="nv">//</span> <span class="nv">Gaussian</span> <span class="nv">Kernel</span> <span class="nv">is:</span></span>
<span id="LC10" class="line">    <span class="nv">//</span> <span class="mi">1</span>  <span class="mi">2</span>  <span class="nv">1</span></span>
<span id="LC11" class="line">    <span class="nv">//</span> <span class="mi">2</span>  <span class="mi">4</span>  <span class="nv">2</span></span>
<span id="LC12" class="line">    <span class="nv">//</span> <span class="mi">1</span>  <span class="mi">2</span>  <span class="nv">1</span></span>
<span id="LC13" class="line">    <span class="nb">float</span> <span class="nv">kernelWeights[9]</span> <span class="nb">=</span> <span class="nv">{</span> <span class="nv">1.0f,</span> <span class="nv">2.0f,</span> <span class="nv">1.0f,</span></span>
<span id="LC14" class="line">                               <span class="nv">2.0f,</span> <span class="nv">4.0f,</span> <span class="nv">2.0f,</span></span>
<span id="LC15" class="line">                               <span class="nv">1.0f,</span> <span class="nv">2.0f,</span> <span class="nv">1.0f</span> <span class="nv">}</span><span class="c1">;</span></span>
<span id="LC16" class="line"></span>
<span id="LC17" class="line">    <span class="nv">int2</span> <span class="nv">startImageCoord</span> <span class="nb">=</span> <span class="p">(</span><span class="nv">int2</span><span class="p">)</span> <span class="p">(</span><span class="nv">get_global_id</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span> <span class="nb">-</span> <span class="mi">1</span><span class="o">,</span> <span class="nv">get_global_id</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span> <span class="nb">-</span> <span class="mi">1</span><span class="p">)</span><span class="c1">;</span></span>
<span id="LC18" class="line">    <span class="nv">int2</span> <span class="nv">endImageCoord</span>   <span class="nb">=</span> <span class="p">(</span><span class="nv">int2</span><span class="p">)</span> <span class="p">(</span><span class="nv">get_global_id</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span> <span class="nb">+</span> <span class="mi">1</span><span class="o">,</span> <span class="nv">get_global_id</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span> <span class="nb">+</span> <span class="mi">1</span><span class="p">)</span><span class="c1">;</span></span>
<span id="LC19" class="line">    <span class="nv">int2</span> <span class="nv">outImageCoord</span> <span class="nb">=</span> <span class="p">(</span><span class="nv">int2</span><span class="p">)</span> <span class="p">(</span><span class="nv">get_global_id</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span><span class="o">,</span> <span class="nv">get_global_id</span><span class="p">(</span><span class="mi">1</span><span class="p">))</span><span class="c1">;</span></span>
<span id="LC20" class="line"></span>
<span id="LC21" class="line">    <span class="k">if</span> <span class="p">(</span><span class="nv">outImageCoord.x</span> <span class="nb">&lt;</span> <span class="nv">width</span> <span class="nv">&amp;&amp;</span> <span class="nv">outImageCoord.y</span> <span class="nb">&lt;</span> <span class="nv">height</span><span class="p">)</span></span>
<span id="LC22" class="line">    <span class="nv">{</span></span>
<span id="LC23" class="line">        <span class="nv">int</span> <span class="nv">weight</span> <span class="nb">=</span> <span class="mi">0</span><span class="c1">;</span></span>
<span id="LC24" class="line">        <span class="nv">float4</span> <span class="nv">outColor</span> <span class="nb">=</span> <span class="p">(</span><span class="nv">float4</span><span class="p">)(</span><span class="nv">0.0f,</span> <span class="nv">0.0f,</span> <span class="nv">0.0f,</span> <span class="nv">0.0f</span><span class="p">)</span><span class="c1">;</span></span>
<span id="LC25" class="line">        <span class="nv">for</span><span class="p">(</span> <span class="nv">int</span> <span class="nv">y</span> <span class="nb">=</span> <span class="nv">startImageCoord.y</span><span class="c1">; y &lt;= endImageCoord.y; y++)</span></span>
<span id="LC26" class="line">        <span class="nv">{</span></span>
<span id="LC27" class="line">            <span class="nv">for</span><span class="p">(</span> <span class="nv">int</span> <span class="nv">x</span> <span class="nb">=</span> <span class="nv">startImageCoord.x</span><span class="c1">; x &lt;= endImageCoord.x; x++)</span></span>
<span id="LC28" class="line">            <span class="nv">{</span></span>
<span id="LC29" class="line">                <span class="nv">outColor</span> <span class="nv">+=</span> <span class="p">(</span><span class="nv">read_imagef</span><span class="p">(</span><span class="nv">srcImg,</span> <span class="nv">sampler,</span> <span class="p">(</span><span class="nv">int2</span><span class="p">)(</span><span class="nv">x,</span> <span class="nv">y</span><span class="p">))</span> <span class="nb">*</span> <span class="p">(</span><span class="nv">kernelWeights[weight]</span> <span class="nb">/</span> <span class="nv">16.0f</span><span class="p">))</span><span class="c1">;</span></span>
<span id="LC30" class="line">                <span class="nv">weight</span> <span class="nv">+=</span> <span class="mi">1</span><span class="c1">;</span></span>
<span id="LC31" class="line">            <span class="nv">}</span></span>
<span id="LC32" class="line">        <span class="nv">}</span></span>
<span id="LC33" class="line"></span>
<span id="LC34" class="line">        <span class="nv">//</span> <span class="nv">Write</span> <span class="k">the</span> <span class="nv">output</span> <span class="nv">value</span> <span class="nv">to</span> <span class="nv">image</span></span>
<span id="LC35" class="line">        <span class="nv">write_imagef</span><span class="p">(</span><span class="nv">dstImg,</span> <span class="nv">outImageCoord,</span> <span class="nv">outColor</span><span class="p">)</span><span class="c1">;</span></span>
<span id="LC36" class="line">    <span class="nv">}</span></span>
<span id="LC37" class="line"><span class="nv">}</span></span></code></pre>
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
<h3 class="page-title">Delete ImageFilter2D.cl</h3>
</div>
<div class="modal-body">
<form class="form-horizontal js-replace-blob-form js-quick-submit js-requires-input" action="/EN-605.417.31_FA16/intro_to_gpu/blob/master/opencl-book-samples/src/Chapter_8/ImageFilter2D/ImageFilter2D.cl" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="_method" value="delete" /><input type="hidden" name="authenticity_token" value="UW8guS24BgkYDuWFEnOx3A10/CHTih2NXq0RR7oB8L0dXoMzZr2XnrwLCWrVI9zvFq0r8ylBPq+hxrDGg8iZqg==" /><div class="form-group commit_message-group">
<label class="control-label" for="commit_message-1c3d5ad2b117bafb9b4960a7ae1fe682">Commit message
</label><div class="col-sm-10">
<div class="commit-message-container">
<div class="max-width-marker"></div>
<textarea name="commit_message" id="commit_message-1c3d5ad2b117bafb9b4960a7ae1fe682" class="form-control js-commit-message" placeholder="Delete ImageFilter2D.cl" required="required" rows="3">
Delete ImageFilter2D.cl</textarea>
</div>
</div>
</div>

<div class="form-group branch">
<label class="control-label" for="target_branch">Target branch</label>
<div class="col-sm-10">
<input type="text" name="target_branch" id="target_branch" value="master" required="required" class="form-control js-target-branch" />
<div class="js-create-merge-request-container">
<div class="checkbox">
<label for="create_merge_request-1e79f0f861307d4cfadaec23fee46695"><input type="checkbox" name="create_merge_request" id="create_merge_request-1e79f0f861307d4cfadaec23fee46695" value="1" class="js-create-merge-request" checked="checked" />
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
<h3 class="page-title">Replace ImageFilter2D.cl</h3>
</div>
<div class="modal-body">
<form class="js-quick-submit js-upload-blob-form form-horizontal" action="/EN-605.417.31_FA16/intro_to_gpu/update/master/opencl-book-samples/src/Chapter_8/ImageFilter2D/ImageFilter2D.cl" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="_method" value="put" /><input type="hidden" name="authenticity_token" value="5GAt6Rvj4RIyDcOiArOPEzDP/4bwyBzfADLqzCzdJIqoUY5jUOZwhZYIL03F4+IgKxYoVAoDP/3/WUtNFRRNnQ==" /><div class="dropzone">
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
<label class="control-label" for="commit_message-6d22530ed25e829726fce1cd6073d8be">Commit message
</label><div class="col-sm-10">
<div class="commit-message-container">
<div class="max-width-marker"></div>
<textarea name="commit_message" id="commit_message-6d22530ed25e829726fce1cd6073d8be" class="form-control js-commit-message" placeholder="Replace ImageFilter2D.cl" required="required" rows="3">
Replace ImageFilter2D.cl</textarea>
</div>
</div>
</div>

<div class="form-group branch">
<label class="control-label" for="target_branch">Target branch</label>
<div class="col-sm-10">
<input type="text" name="target_branch" id="target_branch" value="master" required="required" class="form-control js-target-branch" />
<div class="js-create-merge-request-container">
<div class="checkbox">
<label for="create_merge_request-aa1b1715631c8bd318f10d261db8de2b"><input type="checkbox" name="create_merge_request" id="create_merge_request-aa1b1715631c8bd318f10d261db8de2b" value="1" class="js-create-merge-request" checked="checked" />
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

